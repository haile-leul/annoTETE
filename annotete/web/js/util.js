// Small DOM / UI helpers shared by every view.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(c));
  return el;
}

export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Distinct, stable color per track id (golden angle hue walk).
export function idColor(id, alpha = 1) {
  const hue = (id * 137.508) % 360;
  const light = 52 + (id % 3) * 6;
  return `hsla(${hue.toFixed(1)}, 78%, ${light}%, ${alpha})`;
}

export function toast(msg, kind = '', ms = 2600) {
  const el = h('div', { class: `toast ${kind}` }, msg);
  document.getElementById('toasts').append(el);
  setTimeout(() => el.remove(), ms);
}

export function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function fmtTime(frame, fps) {
  const s = frame / (fps || 25);
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${String(m).padStart(2, '0')}:${r.toFixed(2).padStart(5, '0')}`;
}

// Modal dialog. `build(close)` returns the body element; resolves with the close value.
export function modal(title, build, { wide = false } = {}) {
  return new Promise((resolve) => {
    const root = document.getElementById('modalRoot');
    let bg;
    const close = (v) => { bg.remove(); document.removeEventListener('keydown', onKey, true); resolve(v); };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(null); } };
    const box = h('div', { class: 'modal', style: wide ? { width: 'min(900px, 96vw)' } : null }, h('h3', {}, title));
    box.append(build(close));
    bg = h('div', { class: 'modal-bg', onmousedown: (e) => { if (e.target === bg) close(null); } }, box);
    root.append(bg);
    document.addEventListener('keydown', onKey, true);
    const first = box.querySelector('input, select, button.primary');
    if (first) setTimeout(() => first.focus(), 0);
  });
}

export function prompt(title, label, value = '') {
  return modal(title, (close) => {
    const inp = h('input', { type: 'text', value });
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') close(inp.value); e.stopPropagation(); });
    return h('div', {}, h('label', { class: 'blk' }, label), inp,
      h('div', { class: 'actions' }, h('button', { onclick: () => close(null) }, 'Cancel'),
        h('button', { class: 'primary', onclick: () => close(inp.value) }, 'OK')));
  });
}

export function confirm(title, text, okLabel = 'OK', danger = false) {
  return modal(title, (close) => h('div', {}, h('p', {}, text),
    h('div', { class: 'actions' }, h('button', { onclick: () => close(false) }, 'Cancel'),
      h('button', { class: danger ? 'primary danger' : 'primary', onclick: () => close(true) }, okLabel))));
}

// Context menu: items = [{label, key, act, danger} | '-' | {header}]
export function menu(x, y, items) {
  const m = document.getElementById('menu');
  m.innerHTML = '';
  for (const it of items) {
    if (it === '-') { m.append(h('div', { class: 'sep' })); continue; }
    if (it.header) { m.append(h('div', { class: 'hd' }, it.header)); continue; }
    m.append(h('div', {
      class: `mi ${it.danger ? 'danger' : ''}`,
      onclick: () => { hideMenu(); it.act(); },
    }, h('span', {}, it.label), it.key ? h('kbd', {}, it.key) : ''));
  }
  m.classList.remove('hidden');
  const r = m.getBoundingClientRect();
  m.style.left = `${Math.min(x, innerWidth - r.width - 6)}px`;
  m.style.top = `${Math.min(y, innerHeight - r.height - 6)}px`;
  setTimeout(() => document.addEventListener('mousedown', outside, { once: true }), 0);
  function outside(e) { if (!m.contains(e.target)) hideMenu(); else document.addEventListener('mousedown', outside, { once: true }); }
}
export function hideMenu() { document.getElementById('menu').classList.add('hidden'); }

export const store = {
  get(k, d) { try { const v = localStorage.getItem(`annotete:${k}`); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(`annotete:${k}`, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};

export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
