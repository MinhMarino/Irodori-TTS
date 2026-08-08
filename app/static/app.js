const els = {
  voiceGrid: document.getElementById("voiceGrid"),
  language: document.getElementById("language"),
  text: document.getElementById("text"),
  instruct: document.getElementById("instruct"),
  speakBtn: document.getElementById("speakBtn"),
  sampleBtn: document.getElementById("sampleBtn"),
  charCount: document.getElementById("charCount"),
  player: document.getElementById("player"),
  audio: document.getElementById("audio"),
  download: document.getElementById("download"),
  error: document.getElementById("error"),
  statusPill: document.getElementById("statusPill"),
  statusText: document.getElementById("statusText"),
  modelName: document.getElementById("modelName"),
  deviceName: document.getElementById("deviceName"),
};

const SAMPLES = {
  Auto: "Xin chào! Đây là Irodori TTS — hãy chọn nhân vật và để mình đọc giúp bạn.",
  Chinese: "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
  English: "Hello! Welcome to Irodori TTS. Pick a character and let’s bring your words to life.",
  Japanese: "こんにちは！彩りTTSへようこそ。好きなキャラクターを選んで、文章を音声にしてみましょう。",
  Korean: "안녕하세요! 이로도리 TTS입니다. 캐릭터를 고르고 문장을 음성으로 들어보세요.",
  German: "Hallo! Willkommen bei Irodori TTS. Wähle eine Stimme und lass den Text lebendig werden.",
  French: "Bonjour ! Bienvenue sur Irodori TTS. Choisissez une voix et donnez vie à votre texte.",
  Russian: "Привет! Добро пожаловать в Irodori TTS. Выберите голос и озвучьте свой текст.",
  Portuguese: "Olá! Bem-vindo ao Irodori TTS. Escolha uma voz e dê vida ao seu texto.",
  Spanish: "¡Hola! Bienvenido a Irodori TTS. Elige una voz y da vida a tu texto.",
  Italian: "Ciao! Benvenuto su Irodori TTS. Scegli una voce e dai vita al tuo testo.",
};

let selectedSpeaker = "Vivian";
let speakers = [];
let audioUrl = null;

function setStatus(state, text) {
  els.statusPill.dataset.state = state;
  els.statusText.textContent = text;
}

function showError(msg) {
  if (!msg) {
    els.error.hidden = true;
    els.error.textContent = "";
    return;
  }
  els.error.hidden = false;
  els.error.textContent = msg;
}

function updateCount() {
  els.charCount.textContent = `${els.text.value.length} / 4000`;
}

function renderLanguages(languages) {
  els.language.innerHTML = languages
    .map((l) => `<option value="${l.id}">${l.flag} ${l.label}</option>`)
    .join("");
}

function renderVoices(list) {
  speakers = list;
  els.voiceGrid.innerHTML = "";
  list.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "voice-card";
    btn.style.setProperty("--voice", s.color);
    btn.setAttribute("aria-pressed", String(s.id === selectedSpeaker));
    btn.innerHTML = `
      <div class="voice-top">
        <div class="avatar">${s.emoji}</div>
        <div>
          <strong>${s.name}</strong>
          <small>${s.native} · ${s.gender}</small>
        </div>
      </div>
      <div class="tagline">${s.tagline}</div>
      <p class="desc">${s.description}</p>
    `;
    btn.addEventListener("click", () => {
      selectedSpeaker = s.id;
      els.voiceGrid.querySelectorAll(".voice-card").forEach((el) => {
        el.setAttribute("aria-pressed", "false");
      });
      btn.setAttribute("aria-pressed", "true");

      // Gợi ý ngôn ngữ bản địa của nhân vật nếu đang Auto
      if (els.language.value === "Auto") {
        // giữ Auto — model tự nhận diện
      }
    });
    els.voiceGrid.appendChild(btn);
  });
}

async function loadMeta() {
  const res = await fetch("/api/meta");
  if (!res.ok) throw new Error("Không lấy được metadata");
  const data = await res.json();
  renderLanguages(data.languages);
  renderVoices(data.speakers);
  els.modelName.textContent = data.model;
  els.deviceName.textContent = data.device;
  if (data.ready) setStatus("ready", "Sẵn sàng");
  return data;
}

async function pollHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    els.deviceName.textContent = data.device || "—";
    els.modelName.textContent = data.model || "—";
    if (data.ready) {
      setStatus("ready", "Sẵn sàng");
      return true;
    }
    if (data.error) {
      setStatus("error", "Lỗi tải model");
      showError(data.error);
      return false;
    }
    setStatus("loading", "Đang tải model…");
    return false;
  } catch {
    setStatus("error", "Server offline");
    return false;
  }
}

async function synthesize() {
  showError("");
  const text = els.text.value.trim();
  if (!text) {
    showError("Vui lòng nhập văn bản.");
    return;
  }

  els.speakBtn.disabled = true;
  els.speakBtn.querySelector(".btn-label").textContent = "Đang tổng hợp…";
  els.speakBtn.querySelector(".spinner").hidden = false;

  try {
    const res = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: els.language.value,
        speaker: selectedSpeaker,
        instruct: els.instruct.value.trim(),
      }),
    });

    if (!res.ok) {
      let msg = `Lỗi ${res.status}`;
      try {
        const err = await res.json();
        msg = err.detail || msg;
      } catch {
        msg = await res.text();
      }
      throw new Error(msg);
    }

    const blob = await res.blob();
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    audioUrl = URL.createObjectURL(blob);
    els.audio.src = audioUrl;
    els.download.href = audioUrl;
    els.player.hidden = false;
    await els.audio.play().catch(() => {});
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    els.speakBtn.disabled = false;
    els.speakBtn.querySelector(".btn-label").textContent = "Tạo giọng nói";
    els.speakBtn.querySelector(".spinner").hidden = true;
  }
}

els.text.addEventListener("input", updateCount);
els.sampleBtn.addEventListener("click", () => {
  const lang = els.language.value;
  els.text.value = SAMPLES[lang] || SAMPLES.Auto;
  updateCount();
});
els.speakBtn.addEventListener("click", synthesize);
els.language.addEventListener("change", () => {
  if (!els.text.value.trim()) {
    els.text.value = SAMPLES[els.language.value] || SAMPLES.Auto;
    updateCount();
  }
});

(async function init() {
  updateCount();
  try {
    await loadMeta();
  } catch (err) {
    showError(err.message);
  }

  if (!(await pollHealth())) {
    const timer = setInterval(async () => {
      if (await pollHealth()) clearInterval(timer);
    }, 2500);
  }
})();
