/* ============================================================
   VOGUE VISTA — PROFILE DASHBOARD
   profile.js — Interactions, Animations, AI Chat
   ============================================================ */

(function () {
  'use strict';

  /* ── 1. MOBILE SIDEBAR TOGGLE ── */
  const hamburger = document.getElementById('hamburger');
  const sidebar   = document.getElementById('sidebar');

  if (hamburger && sidebar) {
    hamburger.addEventListener('click', () => {
      const isOpen = sidebar.classList.toggle('open');
      hamburger.classList.toggle('open', isOpen);
      hamburger.setAttribute('aria-expanded', isOpen);
    });

    // Close sidebar when a nav link is tapped on mobile
    sidebar.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        sidebar.classList.remove('open');
        hamburger.classList.remove('open');
      });
    });

    // Close when clicking outside sidebar on mobile
    document.addEventListener('click', (e) => {
      if (
        window.innerWidth <= 720 &&
        sidebar.classList.contains('open') &&
        !sidebar.contains(e.target) &&
        !hamburger.contains(e.target)
      ) {
        sidebar.classList.remove('open');
        hamburger.classList.remove('open');
      }
    });
  }

  /* ── 2. ACTIVE NAV LINK (SCROLL SPY) ── */
  const sections  = document.querySelectorAll('section[id]');
  const navLinks  = document.querySelectorAll('.nav-link[data-section]');

  const observerNav = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          navLinks.forEach(link => {
            link.classList.toggle(
              'active',
              link.dataset.section === entry.target.id
            );
          });
        }
      });
    },
    { threshold: 0.3, rootMargin: '-60px 0px -40% 0px' }
  );

  sections.forEach(s => observerNav.observe(s));

  /* ── 3. INTERSECTION OBSERVER — SECTION FADE IN ── */
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          sectionObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -60px 0px' }
  );

  document.querySelectorAll('.section').forEach(s => sectionObserver.observe(s));

  /* ── 4. ANIMATED STAT COUNTERS ── */
  function animateCounter(el, target, duration = 1400) {
    const start     = performance.now();
    const startVal  = 0;

    function step(timestamp) {
      const elapsed  = timestamp - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const ease     = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(startVal + (target - startVal) * ease);
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }

  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const card   = entry.target;
        const target = parseInt(card.dataset.count, 10) || 0;
        const numEl  = card.querySelector('.counter');
        if (numEl) animateCounter(numEl, target);
        counterObserver.unobserve(card);
      });
    },
    { threshold: 0.5 }
  );

  document.querySelectorAll('.stat-card[data-count]').forEach(c => counterObserver.observe(c));

  /* ── 5. FASHION SCORE RINGS ── */
  // Each <circle.ring-fill> has data-score attribute (0–100)
  // circumference of r=50 circle = 2π×50 ≈ 314.16

  const CIRC = 2 * Math.PI * 50; // 314.159

  // Inject SVG gradient once
  const svgNS  = 'http://www.w3.org/2000/svg';
  const defsEl = document.createElementNS(svgNS, 'defs');
  defsEl.innerHTML = `
    <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#C4A96B"/>
      <stop offset="100%" stop-color="#E8D5A3"/>
    </linearGradient>`;

  const firstRingSvg = document.querySelector('.ring-svg');
  if (firstRingSvg) firstRingSvg.prepend(defsEl);

  const ringObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        const ringEl = entry.target;
        const score  = parseInt(ringEl.dataset.score, 10) || 0;
        const offset = CIRC - (CIRC * score) / 100;
        ringEl.style.stroke = 'url(#goldGradient)';
        ringEl.style.strokeDashoffset = offset;
        ringObserver.unobserve(ringEl);
      });
    },
    { threshold: 0.5 }
  );

  document.querySelectorAll('.ring-fill').forEach(ring => {
    ring.style.strokeDasharray  = CIRC;
    ring.style.strokeDashoffset = CIRC;
    ringObserver.observe(ring);
  });

  /* ── 6. EXPAND / COLLAPSE DETAILS ── */
  document.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.target;
      const detail   = document.getElementById(targetId);
      if (!detail) return;

      const isOpen = detail.classList.toggle('open');
      btn.classList.toggle('open', isOpen);
      btn.querySelector('.expand-icon').textContent = isOpen ? '−' : '+';
    });
  });

  /* ── 7. AI CHAT ── */
  const chatInput    = document.getElementById('chatInput');
  const chatMessages = document.getElementById('chatMessages');
  const sendBtn      = document.getElementById('chatSendBtn');

  // Fashion AI replies (static demo — replace with real API call as needed)
  const fashionResponses = [
    "Based on your colour analysis, I'd recommend warm jewel tones and earthy neutrals. For the occasion you mention, a draped silhouette in deep burgundy or camel would be exquisite.",
    "For a wedding, consider your season palette — if you're an Autumn, rich golds, terracotta, or forest green will be beautifully flattering. Avoid pure white as a guest.",
    "A capsule wardrobe begins with five core pieces: a tailored blazer, high-waist trousers, a silk blouse, a versatile midi dress, and quality denim. Choose in your most flattering neutrals.",
    "Your skin tone analysis suggests you have warm undertones. Earthy tones — camel, rust, olive, and warm coral — will make your complexion glow. Avoid stark cool greys.",
    "The most elegant approach is intentional repetition — choosing a signature silhouette and varying colour and fabric. This is the secret behind every iconic personal style.",
  ];

  let responseIndex = 0;

  function appendBubble(text, type) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${type}-bubble`;
    bubble.innerHTML = `<p>${text}</p>`;
    bubble.style.opacity = '0';
    bubble.style.transform = 'translateY(8px)';
    chatMessages.appendChild(bubble);
    requestAnimationFrame(() => {
      bubble.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
      bubble.style.opacity    = '1';
      bubble.style.transform  = 'translateY(0)';
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
  }

  function appendTyping() {
    const typing = document.createElement('div');
    typing.className = 'chat-bubble assistant-bubble typing-bubble';
    typing.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(typing);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return typing;
  }

  function sendChatMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendBubble(text, 'user');
    chatInput.value = '';
    sendBtn.disabled = true;

    const typingEl = appendTyping();

    setTimeout(() => {
      typingEl.remove();
      const reply = fashionResponses[responseIndex % fashionResponses.length];
      responseIndex++;
      appendBubble(reply, 'assistant');
      sendBtn.disabled = false;
    }, 1200 + Math.random() * 800);
  }

  // Expose globally for inline onclick handlers
  window.sendChatMessage = sendChatMessage;

  if (sendBtn) {
    sendBtn.addEventListener('click', sendChatMessage);
  }
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  // Suggestion chips fill the input
  window.fillSuggestion = function (el) {
    if (chatInput) {
      chatInput.value = el.textContent.trim();
      chatInput.focus();
    }
  };

  /* ── 8. SMOOTH SCROLL FOR NAV LINKS ── */
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const targetId = anchor.getAttribute('href').slice(1);
      const target   = document.getElementById(targetId);
      if (!target) return;
      e.preventDefault();
      const offset = window.innerWidth <= 720 ? 70 : 20;
      const top    = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    });
  });

  /* ── 9. GOLD GLOW ON CARD HOVER (mouse tracking) ── */
  const glowCards = document.querySelectorAll(
    '.stat-card, .score-card, .history-card, .body-card, .wardrobe-card, .timeline-card, .settings-card'
  );

  glowCards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x    = ((e.clientX - rect.left) / rect.width)  * 100;
      const y    = ((e.clientY - rect.top)  / rect.height) * 100;
      card.style.setProperty('--mouse-x', `${x}%`);
      card.style.setProperty('--mouse-y', `${y}%`);
    });
  });

  /* ── 10. STAGGERED CARD ENTRY ── */
  const cardGroups = [
    '.score-grid .score-card',
    '.stats-grid .stat-card',
    '.wardrobe-grid .wardrobe-card',
    '.history-grid .body-card',
    '.settings-grid .settings-card',
  ];

  cardGroups.forEach(selector => {
    const cards = document.querySelectorAll(selector);
    cards.forEach((card, i) => {
      card.style.animationDelay = `${i * 80}ms`;
      card.style.opacity = '0';
      card.style.transform = 'translateY(16px)';
    });

    const staggerObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        entry.target.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        entry.target.style.opacity    = '1';
        entry.target.style.transform  = 'translateY(0)';
        staggerObserver.unobserve(entry.target);
      });
    }, { threshold: 0.1 });

    cards.forEach(card => staggerObserver.observe(card));
  });

  /* ── 11. LOOK CARD QUICK VIEW (placeholder modal) ── */
  document.querySelectorAll('.quick-view-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const card     = btn.closest('.look-card');
      const occasion = card?.querySelector('.look-occasion')?.textContent || 'Look';
      // You can replace this with a real modal implementation
      console.log(`Quick view: ${occasion}`);
      // Example: dispatch a custom event for Django/JS integration
      document.dispatchEvent(new CustomEvent('vv:quickView', {
        detail: { occasion, card }
      }));
    });
  });

  /* ── 12. PAGE LOAD ANIMATION SEQUENCE ── */
  window.addEventListener('load', () => {
    // Trigger the first section immediately
    const hero = document.getElementById('profile-hero');
    if (hero) setTimeout(() => hero.classList.add('visible'), 100);
  });

})();