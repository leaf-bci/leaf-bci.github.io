const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

navToggle?.addEventListener('click', () => {
  const open = navLinks.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(open));
});

document.querySelectorAll('.nav-links a').forEach((link) => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('open');
    navToggle?.setAttribute('aria-expanded', 'false');
  });
});

const abstractBody = document.querySelector('#abstract-body');
const abstractToggle = document.querySelector('#abstract-toggle');

if (abstractBody && abstractToggle) {
  abstractBody.classList.add('is-collapsed');
  abstractToggle.hidden = false;
  abstractToggle.addEventListener('click', () => {
    const expanded = abstractToggle.getAttribute('aria-expanded') === 'true';
    abstractToggle.setAttribute('aria-expanded', String(!expanded));
    abstractToggle.textContent = expanded ? 'Read more' : 'Show less';
    abstractBody.classList.toggle('is-collapsed', expanded);
  });
}

const semanticDemo = document.querySelector('#semantic-demo');

class SemanticClusterPlot {
  constructor(card, data, levels) {
    this.card = card;
    this.data = data;
    this.levels = levels;
    this.canvas = card.querySelector('.cluster-canvas');
    this.wrap = card.querySelector('.cluster-canvas-wrap');
    this.tooltip = card.querySelector('.cluster-tooltip');
    this.context = this.canvas.getContext('2d');
    this.currentCoordinates = data.coordinates[0].map(([x, y]) => [x, y]);
    this.currentLevel = 0;
    this.animationFrame = null;
    this.width = 0;
    this.height = 0;
    this.padding = 24;

    this.populateLegend();
    this.updatePrompt(0);
    this.resize();

    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.wrap);
    } else {
      window.addEventListener('resize', () => this.resize());
    }

    this.canvas.addEventListener('pointermove', (event) => this.showNearestPoint(event));
    this.canvas.addEventListener('pointerleave', () => { this.tooltip.hidden = true; });
  }

  populateLegend() {
    const legend = this.card.querySelector('.cluster-legend');
    this.data.label_names.forEach((name, index) => {
      const item = document.createElement('span');
      const swatch = document.createElement('i');
      swatch.style.backgroundColor = this.data.colors[index];
      item.append(swatch, document.createTextNode(name));
      legend.append(item);
    });
  }

  updatePrompt(level) {
    const prompt = this.data.prompts[level];
    const targets = prompt.candidate_labels.length
      ? ` + [${prompt.candidate_labels.join(', ')}]`
      : '';
    this.card.querySelector('.demo-prompt').textContent = `“${prompt.prompt}”${targets}`;
    this.canvas.setAttribute(
      'aria-label',
      `${this.data.title} embedding clusters under ${this.levels[level].label.toLowerCase()}`,
    );
  }

  resize() {
    const rect = this.wrap.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    this.width = rect.width;
    this.height = rect.height;
    const displayWidth = Math.round(rect.width * pixelRatio);
    const displayHeight = Math.round(rect.height * pixelRatio);
    if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
      this.canvas.width = displayWidth;
      this.canvas.height = displayHeight;
    }
    this.context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    this.draw();
  }

  pointToCanvas([x, y]) {
    const plotWidth = this.width - 2 * this.padding;
    const plotHeight = this.height - 2 * this.padding;
    return [this.padding + x * plotWidth, this.height - this.padding - y * plotHeight];
  }

  drawStar(x, y) {
    const context = this.context;
    const outer = 6.5;
    const inner = 2.8;
    context.beginPath();
    for (let point = 0; point < 10; point += 1) {
      const radius = point % 2 === 0 ? outer : inner;
      const angle = -Math.PI / 2 + point * Math.PI / 5;
      const px = x + Math.cos(angle) * radius;
      const py = y + Math.sin(angle) * radius;
      if (point === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    }
    context.closePath();
    context.fillStyle = '#c23f4b';
    context.fill();
    context.strokeStyle = '#ffffff';
    context.lineWidth = 1.5;
    context.stroke();
  }

  draw() {
    if (!this.width || !this.height) return;
    const context = this.context;
    context.clearRect(0, 0, this.width, this.height);

    this.data.label_names.forEach((_, classIndex) => {
      context.fillStyle = this.data.colors[classIndex];
      if (this.data.sample_count > 2000) {
        context.globalAlpha = 0.5;
        this.currentCoordinates.forEach((point, pointIndex) => {
          if (this.data.point_labels[pointIndex] !== classIndex) return;
          const [x, y] = this.pointToCanvas(point);
          context.fillRect(x - 1.2, y - 1.2, 2.4, 2.4);
        });
      } else {
        context.beginPath();
        this.currentCoordinates.forEach((point, pointIndex) => {
          if (this.data.point_labels[pointIndex] !== classIndex) return;
          const [x, y] = this.pointToCanvas(point);
          context.moveTo(x + 2.35, y);
          context.arc(x, y, 2.35, 0, Math.PI * 2);
        });
        context.globalAlpha = 0.72;
        context.fill();
      }
    });
    context.globalAlpha = 1;

    this.data.prototype_coordinates.forEach((point, index) => {
      const [x, y] = this.pointToCanvas(point);
      this.drawStar(x, y);
      context.font = '800 13px Inter, ui-sans-serif, sans-serif';
      context.textBaseline = 'middle';
      const label = this.data.label_names[index];
      const labelWidth = context.measureText(label).width;
      const placeLeft = x + labelWidth + 12 > this.width - this.padding;
      context.textAlign = placeLeft ? 'right' : 'left';
      const labelX = x + (placeLeft ? -9 : 9);
      context.lineWidth = 3.5;
      context.lineJoin = 'round';
      context.strokeStyle = 'rgba(255, 255, 255, 0.94)';
      context.strokeText(label, labelX, y);
      context.fillStyle = '#b8323f';
      context.fillText(label, labelX, y);
    });
  }

  setLevel(level) {
    if (level === this.currentLevel && !this.animationFrame) {
      this.updatePrompt(level);
      return;
    }
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
    const startCoordinates = this.currentCoordinates.map(([x, y]) => [x, y]);
    const targetCoordinates = this.data.coordinates[level];
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const duration = reduceMotion ? 0 : 680;
    const startTime = performance.now();

    this.currentLevel = level;
    this.updatePrompt(level);
    this.tooltip.hidden = true;

    const step = (now) => {
      const progress = duration === 0 ? 1 : Math.min((now - startTime) / duration, 1);
      const eased = 1 - (1 - progress) ** 3;
      this.currentCoordinates = startCoordinates.map(([startX, startY], index) => {
        const [targetX, targetY] = targetCoordinates[index];
        return [
          startX + (targetX - startX) * eased,
          startY + (targetY - startY) * eased,
        ];
      });
      this.draw();
      if (progress < 1) this.animationFrame = requestAnimationFrame(step);
      else this.animationFrame = null;
    };
    this.animationFrame = requestAnimationFrame(step);
  }

  showNearestPoint(event) {
    const rect = this.canvas.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    let nearest = -1;
    let nearestDistance = 10 ** 2;
    this.currentCoordinates.forEach((point, index) => {
      const [x, y] = this.pointToCanvas(point);
      const distance = (x - pointerX) ** 2 + (y - pointerY) ** 2;
      if (distance < nearestDistance) {
        nearest = index;
        nearestDistance = distance;
      }
    });
    if (nearest < 0) {
      this.tooltip.hidden = true;
      return;
    }
    const labelIndex = this.data.point_labels[nearest];
    this.tooltip.textContent = this.data.label_names[labelIndex];
    this.tooltip.style.left = `${pointerX}px`;
    this.tooltip.style.top = `${pointerY}px`;
    this.tooltip.hidden = false;
  }
}

if (semanticDemo) {
  fetch('assets/data/semantic-guidance-demo.json')
    .then((response) => {
      if (!response.ok) throw new Error(`Semantic demo request failed: ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!Array.isArray(data.levels) || !Array.isArray(data.datasets)) {
        throw new Error('Unexpected semantic demo schema');
      }
      data.datasets.forEach((dataset) => {
        dataset.colors = dataset.colors.map((color) => (
          color.toLowerCase() === '#b64a52' ? '#6f62a6' : color
        ));
      });
      const plots = data.datasets.map((dataset) => {
        const card = semanticDemo.querySelector(`[data-demo-dataset="${dataset.id}"]`);
        if (!card) throw new Error(`Missing semantic demo card: ${dataset.id}`);
        return new SemanticClusterPlot(card, dataset, data.levels);
      });
      const summary = semanticDemo.querySelector('#guidance-summary');
      const buttons = [...semanticDemo.querySelectorAll('[data-guidance-level]')];
      const slider = semanticDemo.querySelector('#guidance-range');
      const pauseButton = semanticDemo.querySelector('#guidance-pause');
      const pauseIcon = pauseButton.querySelector('.guidance-pause-icon');
      const pauseLabel = pauseButton.querySelector('.guidance-pause-label');
      let autoTimer = null;
      let userPaused = false;
      const autoDelay = 2000;

      const stopAutoAdvance = () => {
        window.clearTimeout(autoTimer);
        autoTimer = null;
      };

      const scheduleAutoAdvance = () => {
        stopAutoAdvance();
        if (userPaused || document.hidden) return;
        autoTimer = window.setTimeout(() => {
          const currentLevel = Number(slider.value);
          setGuidanceLevel((currentLevel + 1) % data.levels.length);
        }, autoDelay);
      };

      const setGuidanceLevel = (level) => {
        const selectedLevel = data.levels[level];
        slider.value = String(level);
        slider.style.setProperty('--level-progress', `${level * 50}%`);
        slider.setAttribute('aria-valuetext', selectedLevel.label);
        buttons.forEach((item) => {
          const active = Number(item.dataset.guidanceLevel) === level;
          item.classList.toggle('active', active);
          item.setAttribute('aria-pressed', String(active));
        });
        summary.textContent = selectedLevel.summary;
        plots.forEach((plot) => plot.setLevel(level));
        scheduleAutoAdvance();
      };
      buttons.forEach((button) => {
        button.addEventListener('click', () => {
          setGuidanceLevel(Number(button.dataset.guidanceLevel));
        });
      });
      slider.addEventListener('input', () => setGuidanceLevel(Number(slider.value)));
      pauseButton.addEventListener('click', () => {
        userPaused = !userPaused;
        pauseButton.setAttribute('aria-pressed', String(userPaused));
        pauseButton.setAttribute('aria-label', userPaused ? 'Resume automatic guidance cycle' : 'Pause automatic guidance cycle');
        pauseIcon.textContent = userPaused ? '▶' : 'Ⅱ';
        pauseLabel.textContent = userPaused ? 'Resume' : 'Pause';
        if (userPaused) stopAutoAdvance();
        else scheduleAutoAdvance();
      });
      document.addEventListener('visibilitychange', scheduleAutoAdvance);
      semanticDemo.dataset.state = 'ready';
      scheduleAutoAdvance();
    })
    .catch((error) => {
      semanticDemo.dataset.state = 'error';
      semanticDemo.querySelectorAll('[data-guidance-level]').forEach((button) => {
        button.disabled = true;
      });
      semanticDemo.querySelector('#guidance-range').disabled = true;
      semanticDemo.querySelector('#guidance-pause').disabled = true;
      semanticDemo.querySelector('.demo-error').hidden = false;
      semanticDemo.querySelector('.demo-fallback').hidden = false;
      semanticDemo.querySelectorAll('.cluster-loading').forEach((message) => {
        message.textContent = 'Interactive data unavailable';
      });
      console.error(error);
    });
}

const sections = [...document.querySelectorAll('main section[id]')];
const sectionLinks = new Map(
  [...document.querySelectorAll('.nav-links a')].map((link) => [link.hash.slice(1), link]),
);

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting && sectionLinks.has(entry.target.id))
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    sectionLinks.forEach((link, id) => {
      if (id === visible.target.id) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
    });
  }, { rootMargin: '-20% 0px -65% 0px', threshold: [0, 0.15, 0.5] });
  sections.forEach((section) => observer.observe(section));
}

const resultData = {
  paradigm: {
    kicker: 'Cross-paradigm benchmark',
    title: 'Strong average performance across heterogeneous BCI tasks.',
    description: 'LEAF leads the evaluated baselines across motor imagery, emotion recognition, and the remaining paradigms, demonstrating that one shared language-aligned representation can support diverse label spaces.',
    src: 'assets/figures/paradigm-results.png',
    alt: 'Average balanced accuracy by BCI paradigm',
    width: 1098,
    height: 663,
  },
  instructions: {
    kicker: 'Representation geometry',
    title: 'Instructions compact classes and separate their semantics.',
    description: 'Richer semantic guidance reduces intra-class variation, increases inter-class distance, and organizes EEG embeddings around their textual class prototypes.',
    src: 'assets/figures/instruction-effects.png',
    alt: 'Effect of instruction conditioning on EEG representation geometry',
    width: 1630,
    height: 709,
  },
  zeroshot: {
    kicker: 'Held-out datasets',
    title: 'Language guidance supports transfer without adaptation.',
    description: 'On Dreyer2023 and Weibo2014, LEAF performs direct inference with no labeled adaptation. Task and target semantics improve balanced accuracy, especially when the unconditioned representation is weak.',
    src: 'assets/figures/zero-shot-heldout.png',
    alt: 'Zero-shot direct inference on held-out motor imagery datasets',
    width: 2851,
    height: 957,
  },
  topographies: {
    kicker: 'Spatial interpretation',
    title: 'Task-dependent neural topographies.',
    description: 'Saliency patterns emphasize plausible regions for motor imagery, emotion, SSVEP, covert speech, and workload decoding.',
    src: 'assets/figures/saliency-topographies.png',
    alt: 'Saliency topographies across representative EEG datasets',
    width: 2339,
    height: 426,
  },
};

const resultImage = document.querySelector('#result-image');
const resultLightbox = document.querySelector('#result-lightbox');
const resultKicker = document.querySelector('#result-kicker');
const resultTitle = document.querySelector('#result-title');
const resultDescription = document.querySelector('#result-description');
const resultPanel = document.querySelector('#result-panel');

document.querySelectorAll('.result-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    const data = resultData[tab.dataset.result];
    if (!data) return;

    document.querySelectorAll('.result-tab').forEach((item) => {
      const active = item === tab;
      item.classList.toggle('active', active);
      item.setAttribute('aria-selected', String(active));
    });

    resultKicker.textContent = data.kicker;
    resultTitle.textContent = data.title;
    resultDescription.textContent = data.description;
    resultImage.src = data.src;
    resultImage.alt = data.alt;
    resultImage.width = data.width;
    resultImage.height = data.height;
    resultLightbox.dataset.lightbox = data.src;
    resultPanel.dataset.result = tab.dataset.result;
  });
});

const lightbox = document.querySelector('#lightbox');
const lightboxImage = lightbox?.querySelector('img');

document.addEventListener('click', (event) => {
  const trigger = event.target.closest('[data-lightbox]');
  if (!trigger || !lightbox || !lightboxImage) return;
  lightboxImage.src = trigger.dataset.lightbox;
  lightboxImage.alt = trigger.querySelector('img')?.alt || 'Expanded paper figure';
  lightbox.showModal();
  document.body.classList.add('dialog-open');
});

const closeLightbox = () => {
  if (!lightbox?.open) return;
  lightbox.close();
  document.body.classList.remove('dialog-open');
  lightboxImage.removeAttribute('src');
};

lightbox?.querySelector('.lightbox-close')?.addEventListener('click', closeLightbox);
lightbox?.addEventListener('click', (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox?.addEventListener('cancel', () => {
  document.body.classList.remove('dialog-open');
  lightboxImage.removeAttribute('src');
});

const copyButton = document.querySelector('#copy-citation');
const citationText = document.querySelector('#citation-text');

if (citationText) {
  fetch('assets/jiang2026leaf.bib')
    .then((response) => {
      if (!response.ok) throw new Error(`BibTeX request failed: ${response.status}`);
      return response.text();
    })
    .then((bibtex) => {
      citationText.textContent = bibtex.trim();
    })
    .catch(() => {
      // Keep the embedded BibTeX fallback when the page is opened without a server.
    });
}

copyButton?.addEventListener('click', async () => {
  const citation = citationText?.textContent.trim();
  if (!citation) return;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied';
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
  } catch {
    copyButton.textContent = 'Select text';
  }
});

document.querySelector('#year').textContent = String(new Date().getFullYear());
