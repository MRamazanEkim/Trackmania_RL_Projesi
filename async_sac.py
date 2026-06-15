"""
AsyncSAC — Eşzamanlı toplama + öğrenme SAC
==========================================
SB3 SAC'i, oyun adımı toplamayı (gerçek-zaman, yavaş) ve gradient
öğrenmeyi (CPU, hızlı) AYRI thread'lerde örtüştürecek şekilde sarar.

Neden
-----
Standart SB3 off-policy döngüsü SIRALI:

    while: collect_rollouts (1 oyun adımı, ~50ms boşta bekle) → train (N gradient)

Trackmania gerçek-zaman 20Hz çalıştığı için her adımda ~50ms CPU boşta
bekler, sonra öğrenir. tmrl'in distributed mimarisi bu iki işi paralel
yürütür (RolloutWorker ↔ Trainer). AsyncSAC bunu tek süreçte, iki thread
ile taklit eder:

    • Collector thread : sürekli env.step() → replay buffer (gerçek-zaman burada izole)
    • Ana thread       : buffer doluyken sürekli train() (oyun beklerken CPU öğrenir)

SAC off-policy olduğu için eski/biraz-bayat ağırlıklarla toplanan deneyimden
öğrenmek normaldir — bu, tmrl'in "async actor weights" semantiğiyle aynıdır.

Senkronizasyon
--------------
gradient_step / env_step oranı `max_grad_per_step` ile sınırlanır
(tmrl'in MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP=4 fikri). Böylece öğrenme,
toplamanın çok önüne geçip bayat veriyle aşırı uyum yapmaz.

Thread güvenliği
----------------
• Replay buffer add/sample bir kilitle korunur (kısa numpy işlemleri).
• Collector, AKSİYON ÜRETİMİNDE ana ağı kullanmaz — kendi policy KOPYASINI
  (self._collector_policy) kullanır. Bu kritik: SB3 actor'ün dağılım nesnesi
  (action_dist) stateful'dür; collector ve trainer aynı ağda eşzamanlı forward
  yaparsa bu paylaşılan state bozulur (shape mismatch). Ayrı kopya bunu çözer.
  Kopya, ana ağdan periyodik senkronlanır — tmrl'in "async actor weights"
  semantiğiyle birebir (worker biraz bayat ağırlıkla sürer, bu RL'de normaldir).
• num_timesteps / callback erişimleri ana thread'e bırakılır; collector
  yalnız adım sayacını atomik artırır.

Kullanım
--------
    from async_sac import AsyncSAC
    model = AsyncSAC("MlpPolicy", env, ...)   # SAC ile aynı imza
    model.learn(total_timesteps=...)          # otomatik async
"""

from __future__ import annotations

import threading
import time

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit


class AsyncSAC(SAC):
    """
    SAC + eşzamanlı toplama/öğrenme.

    Ek parametreler
    ---------------
    max_grad_per_step : Toplanan her gerçek oyun adımı başına izin verilen
                        maksimum gradient adımı (oran). tmrl'in
                        MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP karşılığı.
                        Öğrenme bu oranı aşacaksa ana thread, collector
                        yeni adım toplayana dek bekler. Bayat-veri aşırı
                        uyumunu engeller.
    """

    def __init__(self, *args, max_grad_per_step: float = 4.0, sync_every: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_grad_per_step = float(max_grad_per_step)
        # Collector policy kopyasının ana ağdan kaç gradient turunda bir
        # senkronlanacağı. Küçük → daha taze ama daha sık kopya; büyük → daha bayat.
        self._sync_every = int(sync_every)

        # Thread koordinasyonu — learn() içinde kurulur
        self._buffer_lock = threading.Lock()
        self._collector_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Collector tarafından toplanan toplam gerçek adım (atomik sayaç)
        self._collected_steps = 0
        # learn() sırasında callback'in continue_training=False döndüğü an
        self._continue_training = True
        # Collector'ın aksiyon ürettiği ayrı policy kopyası (learn() içinde kurulur)
        self._collector_policy = None
        self._policy_sync_lock = threading.Lock()

    # ── Collector policy (ayrı kopya) ────────────────────────────────────────

    def _sync_collector_policy(self):
        """Ana ağın ağırlıklarını collector kopyasına yaz (state_dict kopyası)."""
        if self._collector_policy is None:
            return
        with self._policy_sync_lock:
            self._collector_policy.load_state_dict(self.policy.state_dict())

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """
        SB3 _sample_action ile aynı; tek fark policy tahmininde ana ağ yerine
        collector KOPYASINI kullanırız (eşzamanlı forward state yarışını önler).
        Warm-up'ta zaten random aksiyon, kopya gerekmez → süper sınıfa düşeriz.
        """
        if self._collector_policy is None or self.num_timesteps < learning_starts:
            return super()._sample_action(learning_starts, action_noise, n_envs)

        from gymnasium import spaces

        assert self._last_obs is not None, "self._last_obs ayarlanmadı"
        with self._policy_sync_lock:
            unscaled_action, _ = self._collector_policy.predict(
                self._last_obs, deterministic=False
            )

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    # ── Collector thread ─────────────────────────────────────────────────────

    def _collector_loop(self, callback, log_interval):
        """
        Ayrı thread: sürekli oyun adımı toplar ve replay buffer'a yazar.
        SB3'ün collect_rollouts mantığını birebir izler, sadece buffer
        yazımını kilitler ve sayacı paylaşır.
        """
        # tek-env, step-bazlı toplama (Trackmania tek oyun penceresi)
        train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
        env = self.env
        assert env is not None

        self.policy.set_training_mode(False)

        try:
            while not self._stop_event.is_set():
                num_collected_steps, num_collected_episodes = 0, 0
                # train_freq=1 → tek adım topla, döngüyü dışarıdan tekrarla
                while should_collect_more_steps(
                    train_freq, num_collected_steps, num_collected_episodes
                ):
                    actions, buffer_actions = self._sample_action(
                        self.learning_starts, self.action_noise, env.num_envs
                    )

                    new_obs, rewards, dones, infos = env.step(actions)

                    self.num_timesteps += env.num_envs
                    num_collected_steps += 1
                    self._collected_steps += env.num_envs

                    # Callback ana thread değil burada çağrılır; SB3 callback'leri
                    # genelde thread-safe değil ama bizim StopOnTimeOrCompletion
                    # yalnız okuma + zaman kontrolü yapıyor → güvenli.
                    callback.update_locals(locals())
                    if not callback.on_step():
                        self._continue_training = False
                        self._stop_event.set()
                        return

                    self._update_info_buffer(infos, dones)

                    with self._buffer_lock:
                        self._store_transition(
                            self.replay_buffer, buffer_actions, new_obs, rewards, dones, infos
                        )

                    self._update_current_progress_remaining(
                        self.num_timesteps, self._total_timesteps
                    )
                    self._on_step()

                    for idx, done in enumerate(dones):
                        if done:
                            num_collected_episodes += 1
                            self._episode_num += 1
                            if self.action_noise is not None:
                                kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                                self.action_noise.reset(**kwargs)
                            if log_interval is not None and self._episode_num % log_interval == 0:
                                self.dump_logs()
        except Exception as exc:  # noqa: BLE001
            # Collector çökerse ana thread'i serbest bırak; hata yüzeye çıksın.
            print(f"[AsyncSAC] Collector thread hatası: {exc}")
            self._continue_training = False
            self._stop_event.set()

    # ── train() — buffer sample'ı kilitle ────────────────────────────────────

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        SB3 SAC.train ile aynı; tek fark replay_buffer.sample çağrısını
        collector'ın add'ı ile yarışmaması için kilitleriz. Gradient hesabı
        kilit dışında kalır (uzun iş, collector'ı bloklamaz).
        """
        # SB3 train() içeride birden çok kez sample çağırır (gradient_steps).
        # Her sample çağrısı kısa; kilidi global sarmak yerine ReplayBuffer.sample'ı
        # geçici olarak kilitli versiyonuyla değiştiriyoruz.
        original_sample = self.replay_buffer.sample

        def _locked_sample(*args, **kwargs):
            with self._buffer_lock:
                return original_sample(*args, **kwargs)

        self.replay_buffer.sample = _locked_sample  # type: ignore[method-assign]
        try:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        finally:
            self.replay_buffer.sample = original_sample  # type: ignore[method-assign]

    # ── learn() — collector thread başlat, ana thread'de öğren ───────────────

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 4,
        tb_log_name: str = "AsyncSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None, "Ortam ayarlanmadan learn() çağrılamaz."

        self._stop_event.clear()
        self._continue_training = True
        self._collected_steps = 0

        # Collector için ana policy'nin derin kopyasını kur (eval modda, grad yok).
        # Aksiyon üretimi bu kopyadan yapılır → trainer'la eşzamanlı forward
        # state yarışı olmaz. Periyodik senkronlanır.
        import copy
        self._collector_policy = copy.deepcopy(self.policy).to(self.device)
        self._collector_policy.set_training_mode(False)
        for p in self._collector_policy.parameters():
            p.requires_grad_(False)
        self._sync_collector_policy()

        # Collector'ı başlat
        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            args=(callback, log_interval),
            name="AsyncSAC-Collector",
            daemon=True,
        )
        self._collector_thread.start()

        grad_steps_done = 0
        try:
            while self.num_timesteps < total_timesteps and self._continue_training:
                # Yeterli warm-up toplanana dek öğrenme; sadece bekle
                if self.num_timesteps <= self.learning_starts or self._collected_steps == 0:
                    time.sleep(0.01)
                    continue

                # Oran sınırı: gradient/adım oranı tavanı aşarsa collector'ın
                # yeni adım toplamasını bekle (tmrl MAX_TRAINING_STEPS_PER_ENV_STEP).
                ratio = grad_steps_done / max(1, self._collected_steps)
                if ratio >= self._max_grad_per_step:
                    time.sleep(0.001)
                    continue

                # Bir tur gradient (gradient_steps kadar). Collector paralel toplar.
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else 1
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                grad_steps_done += gradient_steps

                # Collector kopyasını periyodik senkronla (taze ağırlık).
                if grad_steps_done % self._sync_every < gradient_steps:
                    self._sync_collector_policy()
        except KeyboardInterrupt:
            print("\n[AsyncSAC] Ana thread durduruldu (KeyboardInterrupt).")
        finally:
            # Collector'ı durdur ve bekle
            self._stop_event.set()
            if self._collector_thread is not None:
                self._collector_thread.join(timeout=10.0)

        callback.on_training_end()
        return self
