(() => {
  const root = document.documentElement;
  const header = document.querySelector('[data-header]');
  const menuButton = document.querySelector('[data-menu-toggle]');
  const nav = document.querySelector('[data-nav]');
  const themeButton = document.querySelector('[data-theme-toggle]');
  const themeColorMeta = document.querySelector('meta[name="theme-color"]');
  const toast = document.querySelector('[data-toast]');
  const navLinks = [...document.querySelectorAll('.site-nav a')];
  const mobileNavQuery = window.matchMedia('(max-width: 800px)');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  let savedTheme = null;
  try {
    const candidate = localStorage.getItem('juha-theme');
    if (candidate === 'dark' || candidate === 'light') savedTheme = candidate;
  } catch {
    savedTheme = null;
  }

  const initialTheme = savedTheme || 'light';
  root.dataset.theme = initialTheme;

  const syncThemeUi = () => {
    themeButton?.setAttribute('aria-label', `Switch to ${root.dataset.theme === 'dark' ? 'light' : 'dark'} theme`);
    const backgroundColor = getComputedStyle(root).getPropertyValue('--bg').trim();
    if (backgroundColor) themeColorMeta?.setAttribute('content', backgroundColor);
  };
  syncThemeUi();

  themeButton?.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    try {
      localStorage.setItem('juha-theme', root.dataset.theme);
    } catch {
      // The selected theme still applies for this page view.
    }
    syncThemeUi();
  });

  const menuLabel = menuButton?.querySelector('.sr-only');
  const setMenuState = (open, restoreFocus = false) => {
    const isMobile = mobileNavQuery.matches;
    const nextOpen = isMobile && Boolean(open);

    if (restoreFocus && isMobile && nav?.contains(document.activeElement)) {
      menuButton?.focus({ preventScroll: true });
    }

    menuButton?.setAttribute('aria-expanded', String(nextOpen));
    nav?.classList.toggle('is-open', nextOpen);

    if (isMobile) {
      nav?.setAttribute('aria-hidden', String(!nextOpen));
      nav?.toggleAttribute('inert', !nextOpen);
    } else {
      nav?.removeAttribute('aria-hidden');
      nav?.removeAttribute('inert');
    }

    if (menuLabel) menuLabel.textContent = nextOpen ? 'Close navigation' : 'Open navigation';
  };

  const closeMenu = (restoreFocus = false) => setMenuState(false, restoreFocus);
  setMenuState(false);

  menuButton?.addEventListener('click', () => {
    if (!mobileNavQuery.matches) return;
    const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
    setMenuState(!isOpen);
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && menuButton?.getAttribute('aria-expanded') === 'true') {
      closeMenu(true);
    }
  });

  const syncMenuForViewport = () => setMenuState(false);
  if (typeof mobileNavQuery.addEventListener === 'function') {
    mobileNavQuery.addEventListener('change', syncMenuForViewport);
  } else {
    mobileNavQuery.addListener(syncMenuForViewport);
  }

  const syncHeader = () => header?.classList.toggle('is-scrolled', window.scrollY > 18);
  syncHeader();
  window.addEventListener('scroll', syncHeader, { passive: true });

  const filters = [...document.querySelectorAll('[data-filter]')];
  const publicationList = document.querySelector('[data-publications]');
  const cards = [...(publicationList?.querySelectorAll('[data-topics]') ?? [])];
  const filterStatus = document.querySelector('[data-filter-status]');
  const filterLabels = {
    physiological: 'physiological sensing',
    medical: 'medical imaging',
    segmentation: 'vision segmentation',
    distillation: 'knowledge distillation'
  };

  const cardMatchesFilter = (card, category) => {
    if (category === 'all') return true;
    return (card.dataset.topics || '').split(/\s+/).includes(category);
  };

  const getFilterLabel = (button) => {
    const category = button.dataset.filter;
    if (filterLabels[category]) return filterLabels[category];
    const textNode = [...button.childNodes].find((node) => node.nodeType === 3 && node.textContent.trim());
    return textNode?.textContent.trim() || category || 'selected';
  };

  const updateFilterCounts = () => {
    filters.forEach((button) => {
      const category = button.dataset.filter;
      const count = cards.filter((card) => cardMatchesFilter(card, category)).length;
      const countElement = button.querySelector('span');
      if (countElement) countElement.textContent = String(count);
    });
  };

  const updateFilterStatus = (button, visibleCount, announce) => {
    if (!filterStatus) return;

    const noun = visibleCount === 1 ? 'work' : 'works';
    const label = getFilterLabel(button).toLowerCase();
    const message = button.dataset.filter === 'all'
      ? `Showing all ${visibleCount} ${noun}.`
      : `Showing ${visibleCount} ${label} ${noun}.`;

    if (!announce && filterStatus.textContent === message) return;
    filterStatus.textContent = message;
  };

  const applyFilter = (button, announce = true) => {
    if (!button) return;

    const category = button.dataset.filter;
    filters.forEach((item) => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });

    let visibleCount = 0;
    cards.forEach((card) => {
      const visible = cardMatchesFilter(card, category);
      card.hidden = !visible;
      if (visible) visibleCount += 1;
    });

    updateFilterStatus(button, visibleCount, announce);
  };

  updateFilterCounts();
  filters.forEach((button) => {
    button.addEventListener('click', () => applyFilter(button));
  });
  const initialFilter = filters.find((button) => button.getAttribute('aria-pressed') === 'true') || filters[0];
  applyFilter(initialFilter, false);

  const awardExplorer = document.querySelector('[data-award-explorer]');
  const awardEntries = [...(awardExplorer?.querySelectorAll('[data-award-entry]') ?? [])];
  const awardKeywordButtons = [...(awardExplorer?.querySelectorAll('[data-award-keyword]') ?? [])];
  const awardReset = awardExplorer?.querySelector('[data-award-reset]');
  const awardStatus = awardExplorer?.querySelector('[data-award-status]');
  const awardKeywordLabels = {
    competition: 'Competition',
    national: 'National',
    startup: 'Startup',
    'big-data': 'Big Data',
    'idea-design': 'Idea & Design',
    mentoring: 'Mentoring',
    internship: 'Internship',
    'artificial-intelligence': 'Artificial Intelligence',
    'aws-deepracer': 'AWS DeepRacer',
    'smart-device': 'Smart Device',
    'paper-award': 'Paper Award',
    'medical-imaging': 'Medical Imaging',
    'panoramic-x-ray': 'Panoramic X-ray',
    'in-korean': 'In Korean'
  };

  const getAwardKeywords = (entry) => (entry.dataset.awardKeywords || '')
    .split(/\s+/)
    .filter(Boolean);

  const applyAwardKeyword = (keyword = '', announce = true) => {
    if (!awardExplorer) return;

    if (keyword) {
      awardExplorer.dataset.activeKeyword = keyword;
    } else {
      delete awardExplorer.dataset.activeKeyword;
    }

    awardKeywordButtons.forEach((button) => {
      button.setAttribute('aria-pressed', String(Boolean(keyword) && button.dataset.awardKeyword === keyword));
    });

    let matchingCount = 0;
    awardEntries.forEach((entry) => {
      const matches = Boolean(keyword) && getAwardKeywords(entry).includes(keyword);
      entry.classList.toggle('is-keyword-match', matches);
      entry.classList.toggle('is-keyword-muted', Boolean(keyword) && !matches);
      if (matches) matchingCount += 1;
    });

    if (awardReset) awardReset.disabled = !keyword;
    if (!awardStatus) return;

    const message = keyword
      ? `${awardKeywordLabels[keyword] || keyword} highlights ${matchingCount} of ${awardEntries.length} awards. All awards remain visible.`
      : 'Four keywords are shared by two or more awards. Select one to highlight every match.';

    if (announce || awardStatus.textContent !== message) awardStatus.textContent = message;
  };

  awardKeywordButtons.forEach((button) => {
    button.addEventListener('click', () => {
      const keyword = button.dataset.awardKeyword || '';
      const nextKeyword = awardExplorer?.dataset.activeKeyword === keyword ? '' : keyword;
      applyAwardKeyword(nextKeyword);
    });
  });

  awardExplorer?.querySelectorAll('.award-keyword-global').forEach((button) => {
    const count = awardEntries.filter((entry) => getAwardKeywords(entry).includes(button.dataset.awardKeyword)).length;
    const countElement = button.querySelector('span');
    if (countElement) countElement.textContent = String(count);
  });

  awardReset?.addEventListener('click', () => applyAwardKeyword(''));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && awardExplorer?.dataset.activeKeyword) applyAwardKeyword('');
  });
  applyAwardKeyword('', false);

  const focusAwardTarget = () => {
    if (!window.location.hash.startsWith('#award-')) return;
    let targetId;
    try {
      targetId = decodeURIComponent(window.location.hash.slice(1));
    } catch {
      return;
    }
    const target = document.getElementById(targetId);
    if (target?.matches('[data-award-entry][tabindex="-1"]')) target.focus({ preventScroll: true });
  };

  document.querySelectorAll('.award-jump-link').forEach((link) => {
    link.addEventListener('click', () => window.requestAnimationFrame(focusAwardTarget));
  });
  window.addEventListener('hashchange', focusAwardTarget);
  focusAwardTarget();

  const revealItems = document.querySelectorAll('[data-reveal]');
  if (!reduceMotion && 'IntersectionObserver' in window && revealItems.length) {
    try {
      const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: .08 });

      root.classList.add('reveal-ready');
      revealItems.forEach((item) => revealObserver.observe(item));
    } catch {
      root.classList.remove('reveal-ready');
    }
  }

  const sections = document.querySelectorAll('main section[id]');
  const setCurrentSection = (sectionId) => {
    navLinks.forEach((link) => {
      const current = link.hash === `#${sectionId}`;
      link.classList.toggle('is-current', current);
      if (current) {
        link.setAttribute('aria-current', 'location');
      } else {
        link.removeAttribute('aria-current');
      }
    });
  };

  navLinks.forEach((link) => {
    link.addEventListener('click', () => {
      setCurrentSection(link.hash.slice(1));
      closeMenu(true);
    });
  });

  setCurrentSection(window.location.hash.slice(1));
  window.addEventListener('hashchange', () => setCurrentSection(window.location.hash.slice(1)));

  if ('IntersectionObserver' in window) {
    const sectionObserver = new IntersectionObserver((entries) => {
      const activeEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => {
          const activeLine = window.innerHeight * .35;
          return Math.abs(a.boundingClientRect.top - activeLine) - Math.abs(b.boundingClientRect.top - activeLine);
        })[0];

      if (activeEntry) setCurrentSection(activeEntry.target.id);
    }, { rootMargin: '-30% 0px -60% 0px' });
    sections.forEach((section) => sectionObserver.observe(section));
  }

  let toastTimer = 0;
  let toastClearTimer = 0;
  if (toast) {
    toast.textContent = '';
    toast.classList.remove('is-visible');
  }

  const showToast = (message) => {
    if (!toast) return;

    window.clearTimeout(toastTimer);
    window.clearTimeout(toastClearTimer);
    toast.textContent = message;
    toast.classList.add('is-visible');

    toastTimer = window.setTimeout(() => {
      toast.classList.remove('is-visible');
      toastClearTimer = window.setTimeout(() => {
        toast.textContent = '';
      }, 300);
    }, 2200);
  };

  document.querySelector('[data-copy-email]')?.addEventListener('click', async (event) => {
    const button = event.currentTarget;
    const email = button.dataset.email;
    let copied = false;

    try {
      if (!email || !navigator.clipboard?.writeText) throw new Error('Clipboard API unavailable');
      await navigator.clipboard.writeText(email);
      copied = true;
    } catch {
      let temporary = null;
      try {
        if (!email) throw new Error('Email address unavailable');
        temporary = document.createElement('textarea');
        temporary.value = email;
        temporary.setAttribute('readonly', '');
        temporary.style.position = 'fixed';
        temporary.style.top = '0';
        temporary.style.left = '-9999px';
        temporary.style.opacity = '0';
        document.body.appendChild(temporary);
        temporary.select();
        copied = document.execCommand('copy') === true;
      } catch {
        copied = false;
      } finally {
        temporary?.remove();
        button.focus({ preventScroll: true });
      }
    }

    showToast(copied
      ? 'Email copied to clipboard.'
      : 'Copy failed. Please copy the email address manually.');
  });

  const year = document.querySelector('[data-year]');
  if (year) year.textContent = new Date().getFullYear();
})();
