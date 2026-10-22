(function () {
  var sel = document.querySelectorAll('.filters select');
  var notes = document.getElementById('notes');
  var count = document.querySelector('.filters .count');
  function apply() {
    var f = {};
    sel.forEach(function (s) { f[s.dataset.f] = s.value; });
    var shown = 0, total = 0;
    document.querySelectorAll('tr.run').forEach(function (tr) {
      total++;
      var spec = tr.dataset.spec === 'off' ? 'off' : 'on';
      var ok = (!f.model || tr.dataset.model === f.model) &&
               (!f.engine || tr.dataset.engine === f.engine) &&
               (!f.spec || spec === f.spec);
      tr.hidden = !ok;
      var n = tr.nextElementSibling;
      if (n && n.classList.contains('note')) n.hidden = !(ok && (notes.checked || tr.dataset.open === '1'));
      if (ok) shown++;
    });
    count.textContent = shown + ' of ' + total + ' rows';
  }
  sel.forEach(function (s) { s.addEventListener('change', apply); });
  notes.addEventListener('change', apply);
  document.querySelectorAll('tr.run[data-has-note="true"]').forEach(function (tr) {
    tr.addEventListener('click', function (e) {
      if (e.target.closest('a')) return;
      tr.dataset.open = tr.dataset.open === '1' ? '0' : '1';
      apply();
    });
  });
  document.querySelectorAll('th[data-k]').forEach(function (th) {
    th.addEventListener('click', function () {
      var table = th.closest('table'), tbody = table.tBodies[0];
      var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
      var dir = th.classList.contains('asc') ? -1 : 1;
      table.querySelectorAll('th').forEach(function (h) { h.classList.remove('asc', 'desc'); });
      th.classList.add(dir === 1 ? 'asc' : 'desc');
      var pairs = [];
      Array.prototype.forEach.call(tbody.querySelectorAll('tr.run'), function (tr) {
        var n = tr.nextElementSibling;
        pairs.push([tr, n && n.classList.contains('note') ? n : null]);
      });
      function key(tr) {
        var td = tr.children[idx];
        if (th.dataset.k === 'n') return td.dataset.v === undefined ? -Infinity : parseFloat(td.dataset.v);
        return td.textContent.trim().toLowerCase();
      }
      pairs.sort(function (a, b) {
        var x = key(a[0]), y = key(b[0]);
        return (x < y ? -1 : x > y ? 1 : 0) * dir;
      });
      pairs.forEach(function (p) { tbody.appendChild(p[0]); if (p[1]) tbody.appendChild(p[1]); });
    });
  });
  apply();
})();
