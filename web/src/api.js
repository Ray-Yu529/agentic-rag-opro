// 與 FastAPI 後端溝通的小封裝

export async function startRun(params) {
  const r = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return r.json();
}

export async function getStatus() {
  const r = await fetch("/api/status");
  return r.json();
}

export async function getResults() {
  const r = await fetch("/api/results");
  return r.json();
}

export async function getHistory() {
  const r = await fetch("/api/history");
  return r.json();
}
