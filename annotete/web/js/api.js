// Thin fetch wrappers. Every call rejects with an Error carrying the server message.
async function handle(r) {
  let data = null;
  try { data = await r.json(); } catch (e) { /* non JSON */ }
  if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
  return data;
}

export const get = (url) => fetch(url).then(handle);
export const post = (url, body) => fetch(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
}).then(handle);

export const item = (name) => `/api/item/${encodeURIComponent(name)}`;
