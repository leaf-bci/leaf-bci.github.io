(() => {
  const body = document.body;
  const root = document.documentElement;
  const sidebarToggle = document.querySelector('.sidebar-toggle');
  const sidebar = document.querySelector('.docs-sidebar');
  const backdrop = document.querySelector('.sidebar-backdrop');
  const themeToggle = document.querySelector('.theme-toggle');
  const searchInput = document.querySelector('#doc-search');
  const searchResults = document.querySelector('#search-results');
  let searchIndex = null;

  const closeSidebar = () => {
    body.classList.remove('sidebar-open');
    sidebarToggle?.setAttribute('aria-expanded', 'false');
    if (backdrop) backdrop.hidden = true;
  };

  sidebarToggle?.addEventListener('click', () => {
    const open = !body.classList.contains('sidebar-open');
    body.classList.toggle('sidebar-open', open);
    sidebarToggle.setAttribute('aria-expanded', String(open));
    if (backdrop) backdrop.hidden = !open;
  });

  backdrop?.addEventListener('click', closeSidebar);
  sidebar?.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeSidebar));

  const preferredTheme = () => {
    try {
      const saved = localStorage.getItem('leaf-doc-theme-v2');
      if (saved) return saved;
    } catch (_) {}
    return 'light';
  };

  const setTheme = (theme) => {
    root.dataset.theme = theme;
    themeToggle?.setAttribute('aria-label', `Use ${theme === 'dark' ? 'light' : 'dark'} theme`);
    try { localStorage.setItem('leaf-doc-theme-v2', theme); } catch (_) {}
  };

  setTheme(root.dataset.theme || preferredTheme());
  themeToggle?.addEventListener('click', () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark'));

  document.querySelectorAll('.markdown-body h2[id], .markdown-body h3[id]').forEach((heading) => {
    const anchor = document.createElement('a');
    anchor.className = 'heading-anchor';
    anchor.href = `#${heading.id}`;
    anchor.setAttribute('aria-label', `Link to ${heading.textContent}`);
    anchor.textContent = '#';
    heading.prepend(anchor);
  });

  document.querySelectorAll('.markdown-body pre').forEach((pre) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);
    const button = document.createElement('button');
    button.className = 'copy-code';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      await navigator.clipboard.writeText(pre.innerText);
      button.textContent = 'Copied';
      window.setTimeout(() => { button.textContent = 'Copy'; }, 1200);
    });
    wrapper.appendChild(button);
  });

  document.querySelectorAll('.markdown-body table').forEach((table) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });

  const tocLinks = [...document.querySelectorAll('.page-toc a')];
  if (tocLinks.length && 'IntersectionObserver' in window) {
    const byId = new Map(tocLinks.map((link) => [decodeURIComponent(link.hash.slice(1)), link]));
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      tocLinks.forEach((link) => link.classList.remove('active'));
      byId.get(visible[0].target.id)?.classList.add('active');
    }, { rootMargin: '-78px 0px -72% 0px', threshold: [0, 1] });
    byId.forEach((_, id) => {
      const heading = document.getElementById(id);
      if (heading) observer.observe(heading);
    });
  }

  const closeSearch = () => {
    if (!searchResults) return;
    searchResults.hidden = true;
    searchInput?.setAttribute('aria-expanded', 'false');
  };

  const snippetFor = (text, query) => {
    const compact = text.replace(/\s+/g, ' ').trim();
    const position = compact.toLowerCase().indexOf(query.toLowerCase());
    if (position < 0) return compact.slice(0, 150) + (compact.length > 150 ? '…' : '');
    const start = Math.max(0, position - 55);
    const end = Math.min(compact.length, position + query.length + 90);
    return `${start > 0 ? '…' : ''}${compact.slice(start, end)}${end < compact.length ? '…' : ''}`;
  };

  const renderSearch = (query) => {
    if (!searchResults || !searchIndex) return;
    searchResults.replaceChildren();
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      closeSearch();
      return;
    }

    const terms = normalized.split(/\s+/).filter(Boolean);
    const matches = searchIndex
      .map((entry) => {
        const haystack = `${entry.title} ${entry.group} ${entry.text}`.toLowerCase();
        const score = terms.reduce((total, term) => total + (entry.title.toLowerCase().includes(term) ? 5 : 0) + (haystack.includes(term) ? 1 : -20), 0);
        return { entry, score };
      })
      .filter(({ score }) => score >= 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);

    if (!matches.length) {
      const empty = document.createElement('p');
      empty.className = 'search-empty';
      empty.textContent = `No results for “${query.trim()}”`;
      searchResults.appendChild(empty);
    } else {
      const docsHome = new URL(body.dataset.docsHome, window.location.href);
      matches.forEach(({ entry }) => {
        const link = document.createElement('a');
        link.className = 'search-result';
        link.href = new URL(entry.url || './', docsHome).href;
        const group = document.createElement('span');
        group.textContent = entry.group;
        const title = document.createElement('strong');
        title.textContent = entry.title;
        const snippet = document.createElement('p');
        snippet.textContent = snippetFor(entry.text, query.trim());
        link.append(group, title, snippet);
        searchResults.appendChild(link);
      });
    }
    searchResults.hidden = false;
    searchInput?.setAttribute('aria-expanded', 'true');
  };

  const loadSearch = async () => {
    if (searchIndex) return searchIndex;
    const response = await fetch(body.dataset.searchIndex);
    if (!response.ok) throw new Error(`Search index request failed: ${response.status}`);
    searchIndex = await response.json();
    return searchIndex;
  };

  searchInput?.addEventListener('input', async () => {
    try {
      await loadSearch();
      renderSearch(searchInput.value);
    } catch (error) {
      console.error(error);
    }
  });

  searchInput?.addEventListener('focus', async () => {
    try {
      await loadSearch();
      if (searchInput.value) renderSearch(searchInput.value);
    } catch (error) {
      console.error(error);
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === '/' && document.activeElement !== searchInput && !/INPUT|TEXTAREA/.test(document.activeElement?.tagName || '')) {
      event.preventDefault();
      searchInput?.focus();
    }
    if (event.key === 'Escape') {
      closeSearch();
      closeSidebar();
      searchInput?.blur();
    }
  });

  document.addEventListener('click', (event) => {
    if (!event.target.closest('.search-box') && !event.target.closest('.search-results')) closeSearch();
  });
})();
