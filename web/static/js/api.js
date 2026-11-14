// Thin fetch wrapper: always returns parsed JSON or throws an Error with a readable message.
export async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  let res;
  try { res = await fetch(path, opts); } catch (e) { throw new Error("Network error: server unreachable"); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON body */ }
  if (!res.ok) {
    let msg = (data && data.detail) || res.statusText;
    if (Array.isArray(msg)) msg = msg.map((m) => (m.loc ? m.loc.slice(1).join(".") + ": " : "") + m.msg).join("; ");
    throw new Error(msg);
  }
  return data;
}

export function debounce(fn, ms = 120) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
