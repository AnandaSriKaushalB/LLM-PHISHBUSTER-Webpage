// Elements
const analyzeBtn = document.getElementById('analyzeBtn');
const subjectEl = document.getElementById('subject');
const bodyEl = document.getElementById('body');
const statusPill = document.getElementById('statusPill');
const statusMain = document.getElementById('statusMain');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const lastChecked = document.getElementById('lastChecked');
const apiStatus = document.getElementById('apiStatus');
const historyEl = document.getElementById('history');
const sampleBtn = document.getElementById('sampleBtn');
const samplePhishBtn = document.getElementById('samplePhishBtn');
const sampleSuspiciousBtn = document.getElementById('sampleSuspiciousBtn');

const behavioralCard = document.getElementById('behavioralCard');
const behavioralTable = document.getElementById('behavioralTable').querySelector('tbody');

// API Health Check
async function checkAPI() {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "health check" })
    });
    apiStatus.textContent = res.ok ? "API: Connected ✅" : "API: Error ❌";
  } catch {
    apiStatus.textContent = "API: Offline 🔴";
  }
}
checkAPI();

// Handle Analyze Button
analyzeBtn.onclick = async () => {
  const subject = subjectEl.value.trim();
  const body = bodyEl.value.trim();
  const text = (subject + " " + body).trim();
  if (!text) return alert("Please paste the email text.");

  apiStatus.textContent = "Analyzing...";
  statusMain.textContent = "Processing your email content...";
  confidenceFill.style.width = "0%";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    if (!res.ok) {
      statusMain.textContent = data.message || "Server error occurred.";
      apiStatus.textContent = "API: Error ❌";
      return;
    }

    const pred = data.prediction || 'Safe';
    const conf = data.confidence || 50;
    const behaviors = data.behavioral || {};

    updateStatus(pred, conf);
    updateBehavioralTable(behaviors);
    addHistory(subject || body.slice(0, 80), pred, conf);

    apiStatus.textContent = "API: Ready ✅";
  } catch {
    apiStatus.textContent = "API: Request failed ⚠️";
    statusMain.textContent = "Error: Unable to connect to the server.";
  }
};

// Update Status Card UI
function updateStatus(pred, conf) {
  const percent = Math.round(Number(conf) || 0);
  confidenceFill.style.width = percent + '%';
  confidenceText.textContent = `Confidence: ${percent}%`;
  lastChecked.textContent = new Date().toLocaleString();
  statusPill.classList.remove('blink');

  if (pred === 'Fraudulent') {
    statusPill.textContent = 'Fraudulent';
    statusPill.style.background = 'linear-gradient(90deg, crimson, red)';
    statusPill.classList.add('blink');
    statusMain.innerHTML = `<strong style="color:#ff9b9b">Fraudulent email detected!</strong> This email poses a high phishing risk.`;
  } else if (pred === 'Suspicious') {
    statusPill.textContent = 'Suspicious';
    statusPill.style.background = 'linear-gradient(90deg, gold, orange)';
    statusPill.classList.add('blink');
    statusMain.innerHTML = `<strong style="color:#ffd580">Suspicious email detected!</strong> Proceed with caution and verify the sender.`;
  } else {
    statusPill.textContent = 'Safe';
    statusPill.style.background = 'linear-gradient(90deg, limegreen, seagreen)';
    statusPill.classList.add('blink');
    statusMain.innerHTML = `<strong style="color:#9effc4">This email appears safe.</strong> No immediate phishing indicators detected.`;
  }
}

// Helper: infer proper unit
function getUnitForParam(param, value) {
  const lowerParam = param.toLowerCase();

  if (lowerParam.includes('percent') || lowerParam.includes('prob') || lowerParam.includes('confidence')) {
    return '%';
  } else if (lowerParam.includes('time') || lowerParam.includes('duration') || lowerParam.includes('delay')) {
    return ' sec';
  } else if (lowerParam.includes('count') || lowerParam.includes('number') || lowerParam.includes('attempts')) {
    return ' count';
  } else if (lowerParam.includes('score')) {
    return ' pts';
  } else if (lowerParam.includes('ratio')) {
    return ' ratio';
  } else if (typeof value === 'number' && value < 1 && value > 0) {
    return ' (probability)';
  }
  return ''; // no unit for text or other params
}

// Behavioral Metrics Table
function updateBehavioralTable(behaviors) {
  if (!behaviors || Object.keys(behaviors).length === 0) {
    behavioralCard.style.display = "none";
    return;
  }

  behavioralCard.style.display = "block";
  behavioralTable.innerHTML = "";

  const excludeKeys = ["analysis_summary", "category"];

  for (const [param, val] of Object.entries(behaviors)) {
    if (excludeKeys.includes(param)) continue;

    const row = document.createElement('tr');
    const formattedParam = param
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());

    let displayVal;
    if (typeof val === 'number') {
      const unit = getUnitForParam(param, val);
      displayVal = val.toFixed(2) + unit;
    } else {
      displayVal = val;
    }

    row.innerHTML = `<td><strong>${formattedParam}</strong></td><td>${displayVal}</td>`;
    behavioralTable.appendChild(row);
  }

  const separator = document.createElement('tr');
  separator.innerHTML = `<td colspan="2"><hr></td>`;
  behavioralTable.appendChild(separator);

  const summaryRow = document.createElement('tr');
  summaryRow.innerHTML = `
  <td colspan="2">
    <strong>📊 Mail Analysis Summary:</strong><br>
    <pre style="font-family:inherit; margin:4px 0;">${behaviors.analysis_summary.replace(/\n/g, '<br>')}</pre>
  </td>`;
  behavioralTable.appendChild(summaryRow);
}

// History Section
function addHistory(title, pred, conf) {
  const div = document.createElement('div');
  const index = historyEl.children.length + 1;

  // Get formatted timestamp
  const timestamp = new Date().toLocaleString('en-IN', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });

  div.className = 'history-item';
  div.innerHTML = `
    <div><strong>${index}. ${pred}</strong> — ${Math.round(conf)}%</div>
    <div class="muted">${(title || '').slice(0, 90)}${(title || '').length > 90 ? '...' : ''}</div>
    <div class="timestamp">🕒 ${timestamp}</div>
  `;
  historyEl.prepend(div);

  // Re-number all items
  Array.from(historyEl.children).forEach((child, i) => {
    const firstLine = child.querySelector('strong');
    if (firstLine) {
      const text = firstLine.textContent.replace(/^\d+\.\s*/, '');
      firstLine.textContent = `${i + 1}. ${text}`;
    }
  });

  // Keep only latest 6 entries
  while (historyEl.children.length > 6) {
    historyEl.removeChild(historyEl.lastChild);
  }
}


// Sample Buttons
sampleBtn.onclick = () => {
  subjectEl.value = "Weekly team meeting reminder";
  bodyEl.value = "Hi team, reminder about our sync tomorrow at 2 PM. Agenda attached.";
};

samplePhishBtn.onclick = () => {
  subjectEl.value = "URGENT: Verify your bank account now";
  bodyEl.value = "Your account will be suspended! Click here to verify: http://bank-verify123.biz";
};

sampleSuspiciousBtn.onclick = () => {
  subjectEl.value = "Notice: Unusual sign-in attempt";
  bodyEl.value = "We detected a sign-in to your account from a new device. If this wasn't you, please update your account details at: http://account-security-check.example.com/login\n\nIf you didn't initiate this, ignore this message or contact support.";
};
