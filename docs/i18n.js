/* ═══════════════════════════════════════════════════
   MythosForge — Internationalization (i18n)
   Z.AI Research Lab — April 2026
   ═══════════════════════════════════════════════════ */

const translations = {
  es: {
    // Nav
    "nav.architecture": "Arquitectura",
    "nav.components": "Componentes",
    "nav.validation": "Validación",
    "nav.guide": "Guía",
    "nav.files": "Archivos",
    "nav.roadmap": "Ruta",
    "nav.refs": "Bibliografía",
    "nav.contribute": "Contribuir",

    // Hero
    "hero.badge1": "Z.AI Research Lab",
    "hero.badge2": "Open Source",
    "hero.title1": "Mythos",
    "hero.title2": "Forge",
    "hero.desc": 'Laboratorio de investigación sobre <strong>Recurrent-Depth Transformers</strong>: razonamiento latente, MoE, atención MLA/GQA y halting adaptativo.',
    "hero.btn1": "Explorar Arquitectura",
    "hero.btn2": "Guía Rápida",
    "hero.disclaimer": '<span>OpenMythos se presenta como <strong>reconstrucción teórica independiente</strong>, no afiliada a Anthropic. Este laboratorio no afirma equivalencia con ningún sistema propietario.</span>',

    // Architecture
    "arch.sectionBadge": "Arquitectura",
    "arch.sectionTitle": "Flujo Operativo",
    "arch.sectionDesc": "La profundidad útil no viene de apilar capas distintas, sino de aplicar varias veces el mismo bloque compartido durante un único forward pass.",
    "arch.input": "Input Tokens",
    "arch.prelude": "PRELUDE",
    "arch.preludeSub": "Bloque Transformer denso · FFN SwiGLU · 1× ejecución",
    "arch.recurrent": "RECURRENT BLOCK ×N",
    "arch.recurrentSub": "TransformerBlock compartido · Loop × max_loop_iters",
    "arch.chip1.title": "MLA / GQA",
    "arch.chip1.sub": "Atención conmutable",
    "arch.chip2.title": "MoE",
    "arch.chip2.sub": "Routed + Shared Experts",
    "arch.chip3.title": "LoRA",
    "arch.chip3.sub": "Adaptador por iteración",
    "arch.chip4.title": "LTI",
    "arch.chip4.sub": "Inyección estable",
    "arch.act": "ACT — Adaptive Computation Time (Halting por posición)",
    "arch.coda": "CODA",
    "arch.codaSub": "Bloque Transformer denso · FFN SwiGLU · Proyección final",
    "arch.rmsnorm": "RMSNorm",
    "arch.lmhead": "LM Head",
    "arch.output": "Output Logits",
    "arch.loop": "LOOP",
    "arch.infoTitle": "Regla Recurrente:",
    "arch.info": "h<sub>(t+1)</sub> = A · h<sub>t</sub> + B · e + Transformer(h<sub>t</sub>, e) — El razonamiento intermedio se acumula en el estado oculto sin expresarse como tokens visibles.",

    // Components
    "comp.sectionBadge": "Componentes",
    "comp.sectionTitle": "Piezas Arquitectónicas",
    "comp.sectionDesc": "Cada componente seleccionado por coherencia con la literatura más reciente en transformers recurrentes y razonamiento latente.",
    "comp.mla.title": "Atención Conmutable MLA / GQA",
    "comp.mla.desc": "GQA reduce caché KV compartiendo pares K/V. MLA comprime la ruta KV en un latente de bajo rango. Conmutable vía configuración.",
    "comp.moe.title": "MoE en Bloque Recurrente",
    "comp.moe.desc": "Routed experts top-K + shared experts siempre activos. Hereda de DeepSeekMoE la segmentación fina y aislamiento de expertos compartidos.",
    "comp.lti.title": "Inyección LTI Estable",
    "comp.lti.desc": "Estabilidad espectral con A ∈ (0,1) por parametrización log-space. Parche numérico incluido para evitar saturación float32 en el borde.",
    "comp.act.title": "ACT — Halting Adaptativo",
    "comp.act.desc": "Adaptive Computation Time aprende probabilidad de halting por posición. Algunas posiciones dejan de acumular actualizaciones antes que otras.",
    "comp.lora.title": "Adaptador LoRA por Iteración",
    "comp.lora.desc": "Cada iteración modifica su comportamiento con un adaptador LoRA dependiente de la iteración. Enriquece sin multiplicar parámetros estáticos.",
    "comp.prelude.title": "Prelude + Recurrent + Coda",
    "comp.prelude.desc": "Arquitectura de tres etapas: bloques densos de entrada y salida ejecutados una vez, con bloque recurrente compartido aplicado N veces.",

    // Validation
    "val.sectionBadge": "Validación",
    "val.sectionTitle": "Pruebas Ejecutadas",
    "val.sectionDesc": "Validación práctica en CPU con PyTorch 2.10.0+cpu. Configuraciones pequeñas verificadas.",
    "val.col1Title": "Chequeos de Inferencia",
    "val.check1": "Import y construcción del modelo",
    "val.check2": "Forward mínimo con GQA",
    "val.check3": "Forward mínimo con MLA",
    "val.check4": "Generación mínima autoregresiva",
    "val.check5": "Matriz A de LTI en rango estable",
    "val.check6": "test_spectral_radius_stable_after_large_grad_step",
    "val.check6Note": "Fallo numérico float32 en borde 1.0 — parche disponible",
    "val.col2Title": "Estado del Repositorio Original",
    "val.repo1": "Repositorio público",
    "val.repo2": "Licencia MIT",
    "val.repo3": "Implementación PyTorch",
    "val.repo4": "Ejemplo mínimo de uso",
    "val.repo5": "Pruebas automatizadas",
    "val.repo6": "Pesos preentrenados",
    "val.repo7": "Tokenizer propio",
    "val.repo8": "Pipeline entrenamiento completo",
    "val.repo9": "Equivalencia Claude Mythos",
    "val.yes": "SI",
    "val.no": "NO",
    "val.findingTitle": "Hallazgo:",
    "val.finding": '<code>test_spectral_radius_stable_after_large_grad_step</code> falla por redondeo float32. Parche propuesto resuelve sin alterar la arquitectura.',

    // Guide
    "guide.sectionBadge": "Guía Rápida",
    "guide.sectionTitle": "Replicar y Ejecutar Hoy",
    "guide.sectionDesc": "Procedimiento mínimo sin adornos ni suposiciones sobre lo que el repositorio todavía no ofrece.",
    "guide.step1": "1. Preparación del entorno",
    "guide.step2": "2. Arranque rápido",
    "guide.step3": "3. Validación",
    "guide.step4": "4. Qué debes esperar",
    "guide.expect1": "El modelo construye embeddings, RoPE, prelude, recurrente, coda y LM head sin error.",
    "guide.expect2": "El forward devuelve logits con forma (B, T, vocab_size).",
    "guide.expect3": "generate añade tokens y reutiliza caché KV correctamente.",
    "guide.expect4": "La variante MLA es más pesada conceptualmente, pero funcional en configuración pequeña.",
    "guide.expect5": "La matriz A de LTI permanece en rango estable bajo condiciones normales.",

    // Files
    "files.sectionBadge": "Archivos",
    "files.sectionTitle": "Recursos Descargables",
    "files.sectionDesc": "Guía técnica, script de verificación y parche de estabilidad LTI.",
    "files.guide": "Guía Técnica 2026",
    "files.guidePdf": "PDF — Documento completo",
    "files.guideDocx": "DOCX — Versión editable",
    "files.quickstart": "Quickstart Script",
    "files.quickstartDesc": "Python — Verificación mínima",
    "files.patch": "Parche LTI",
    "files.patchDesc": "Diff — Estabilidad numérica",

    // Roadmap
    "road.sectionBadge": "Hoja de Ruta",
    "road.sectionTitle": "De Prototipo a Banco de Pruebas",
    "road.sectionDesc": "Demostrar overfitting, pérdida estable y comportamiento predecible al aumentar n_loops antes de escalar.",
    "road.phase0": "FASE 0",
    "road.phase0Title": "Congelar versión",
    "road.phase0Desc": "Fork del repositorio, etiqueta de commit y requirements-lock para reproducibilidad total.",
    "road.phase1": "FASE 1",
    "road.phase1Title": "Corregir estabilidad operativa",
    "road.phase1Desc": "Aplicar parche LTI para evitar que A alcance exactamente 1.0 en float32.",
    "road.phase2": "FASE 2",
    "road.phase2Title": "Añadir empaquetado",
    "road.phase2Desc": "pyproject.toml, instalación editable y CLI simple.",
    "road.phase3": "FASE 3",
    "road.phase3Title": "Tokenizer y datos",
    "road.phase3Desc": "Definir tokenizer, dataset y formato causal LM.",
    "road.phase4": "FASE 4",
    "road.phase4Title": "Entrenamiento mínimo",
    "road.phase4Desc": "Script train.py con warmup, clipping, checkpoints y logging.",
    "road.phase5": "FASE 5",
    "road.phase5Title": "Benchmarks",
    "road.phase5Desc": "Curvas vs loops, comparación GQA vs MLA, ablations MoE/LoRA/ACT.",
    "road.phase6": "FASE 6",
    "road.phase6Title": "Publicación y comparativa",
    "road.phase6Desc": "Comparativas contra transformers densos de igual presupuesto de FLOPs.",
    "road.done": "Completada",

    // Contribute
    "contribute.sectionBadge": "Contribuir",
    "contribute.sectionTitle": "Únete al Laboratorio",
    "contribute.sectionDesc": "MythosForge es un proyecto abierto. Hay muchas formas de colaborar, desde experimentos hasta documentación.",
    "contribute.issues.title": "Reporta Issues",
    "contribute.issues.desc": "Encuentra bugs, errores en la documentación o mejoras posibles. Abre un issue con contexto claro y pasos para reproducir.",
    "contribute.prs.title": "Envía Pull Requests",
    "contribute.prs.desc": "Parches de código, mejoras de arquitectura, nuevos experimentos o integraciones. Lee la guía CONTRIBUTING.md antes.",
    "contribute.discuss.title": "Discute Ideas",
    "contribute.discuss.desc": "Propón experimentos, comparte resultados o debate sobre la arquitectura RDT en las discusiones del repositorio.",

    // Bibliography
    "refs.sectionBadge": "Bibliografía",
    "refs.sectionTitle": "Referencias Académicas",
    "refs.sectionDesc": "Trabajos clave que sostienen la hipótesis técnica.",

    // Footer
    "footer.brand": '<span class="text-cyan">Mythos</span><span class="text-gold">Forge</span> — Creado por <a href="https://github.com/smouj" target="_blank" rel="noopener" style="color:var(--cyan)">Smouj013</a> con agentes de IA',
    "footer.copy": "Documento técnico independiente — Abril 2026",
  },

  en: {
    // Nav
    "nav.architecture": "Architecture",
    "nav.components": "Components",
    "nav.validation": "Validation",
    "nav.guide": "Guide",
    "nav.files": "Files",
    "nav.roadmap": "Roadmap",
    "nav.refs": "Bibliography",
    "nav.contribute": "Contribute",

    // Hero
    "hero.badge1": "Z.AI Research Lab",
    "hero.badge2": "Open Source",
    "hero.title1": "Mythos",
    "hero.title2": "Forge",
    "hero.desc": 'Research lab on <strong>Recurrent-Depth Transformers</strong>: latent reasoning, MoE, switchable MLA/GQA attention and adaptive halting.',
    "hero.btn1": "Explore Architecture",
    "hero.btn2": "Quickstart Guide",
    "hero.disclaimer": '<span>OpenMythos is presented as an <strong>independent theoretical reconstruction</strong>, not affiliated with Anthropic. This lab does not claim equivalence with any proprietary system.</span>',

    // Architecture
    "arch.sectionBadge": "Architecture",
    "arch.sectionTitle": "Operational Flow",
    "arch.sectionDesc": "Useful depth doesn't come from stacking different layers — it comes from applying the same shared block multiple times during a single forward pass.",
    "arch.input": "Input Tokens",
    "arch.prelude": "PRELUDE",
    "arch.preludeSub": "Dense Transformer block · FFN SwiGLU · 1× execution",
    "arch.recurrent": "RECURRENT BLOCK ×N",
    "arch.recurrentSub": "Shared TransformerBlock · Loop × max_loop_iters",
    "arch.chip1.title": "MLA / GQA",
    "arch.chip1.sub": "Switchable attention",
    "arch.chip2.title": "MoE",
    "arch.chip2.sub": "Routed + Shared Experts",
    "arch.chip3.title": "LoRA",
    "arch.chip3.sub": "Per-iteration adapter",
    "arch.chip4.title": "LTI",
    "arch.chip4.sub": "Stable injection",
    "arch.act": "ACT — Adaptive Computation Time (Per-position halting)",
    "arch.coda": "CODA",
    "arch.codaSub": "Dense Transformer block · FFN SwiGLU · Final projection",
    "arch.rmsnorm": "RMSNorm",
    "arch.lmhead": "LM Head",
    "arch.output": "Output Logits",
    "arch.loop": "LOOP",
    "arch.infoTitle": "Recurrent Rule:",
    "arch.info": "h<sub>(t+1)</sub> = A · h<sub>t</sub> + B · e + Transformer(h<sub>t</sub>, e) — Intermediate reasoning accumulates in the hidden state without being expressed as visible tokens.",

    // Components
    "comp.sectionBadge": "Components",
    "comp.sectionTitle": "Architectural Pieces",
    "comp.sectionDesc": "Each component selected for coherence with the latest literature on recurrent transformers and latent reasoning.",
    "comp.mla.title": "Switchable MLA / GQA Attention",
    "comp.mla.desc": "GQA reduces KV cache by sharing K/V pairs across query head groups. MLA compresses the KV route into a low-rank latent. Switchable via configuration.",
    "comp.moe.title": "MoE in Recurrent Block",
    "comp.moe.desc": "Routed top-K experts + always-active shared experts. Inherits fine-grained segmentation and shared expert isolation from DeepSeekMoE.",
    "comp.lti.title": "Stable LTI Injection",
    "comp.lti.desc": "Spectral stability with A ∈ (0,1) via log-space parametrization. Numeric patch included to prevent float32 saturation at the boundary.",
    "comp.act.title": "ACT — Adaptive Halting",
    "comp.act.desc": "Adaptive Computation Time learns a halting probability per sequence position. Some positions stop accumulating updates earlier than others.",
    "comp.lora.title": "Per-Iteration LoRA Adapter",
    "comp.lora.desc": "Each iteration modifies its behavior with an iteration-dependent LoRA adapter. Enriches without multiplying static parameters.",
    "comp.prelude.title": "Prelude + Recurrent + Coda",
    "comp.prelude.desc": "Three-stage architecture: input and output dense blocks executed once, with a shared recurrent block applied N times.",

    // Validation
    "val.sectionBadge": "Validation",
    "val.sectionTitle": "Executed Tests",
    "val.sectionDesc": "Practical validation on CPU with PyTorch 2.10.0+cpu. Small configurations verified.",
    "val.col1Title": "Inference Checks",
    "val.check1": "Model import and construction",
    "val.check2": "Minimal forward with GQA",
    "val.check3": "Minimal forward with MLA",
    "val.check4": "Minimal autoregressive generation",
    "val.check5": "LTI A matrix in stable range",
    "val.check6": "test_spectral_radius_stable_after_large_grad_step",
    "val.check6Note": "float32 numeric failure at 1.0 boundary — patch available",
    "val.col2Title": "Original Repository Status",
    "val.repo1": "Public repository",
    "val.repo2": "MIT License",
    "val.repo3": "PyTorch implementation",
    "val.repo4": "Minimal usage example",
    "val.repo5": "Automated tests",
    "val.repo6": "Pretrained weights",
    "val.repo7": "Custom tokenizer",
    "val.repo8": "Complete training pipeline",
    "val.repo9": "Claude Mythos equivalence",
    "val.yes": "YES",
    "val.no": "NO",
    "val.findingTitle": "Finding:",
    "val.finding": '<code>test_spectral_radius_stable_after_large_grad_step</code> fails due to float32 rounding. Proposed patch resolves without altering the architecture.',

    // Guide
    "guide.sectionBadge": "Quickstart",
    "guide.sectionTitle": "Replicate and Run Today",
    "guide.sectionDesc": "Minimal procedure with no assumptions about what the repository doesn't yet provide.",
    "guide.step1": "1. Environment setup",
    "guide.step2": "2. Quick start",
    "guide.step3": "3. Validation",
    "guide.step4": "4. What to expect",
    "guide.expect1": "The model builds embeddings, RoPE, prelude, recurrent, coda and LM head without errors.",
    "guide.expect2": "The forward pass returns logits with shape (B, T, vocab_size).",
    "guide.expect3": "generate adds tokens and correctly reuses KV cache.",
    "guide.expect4": "The MLA variant is conceptually heavier, but functional in small configuration.",
    "guide.expect5": "The LTI A matrix remains in stable range under normal conditions.",

    // Files
    "files.sectionBadge": "Files",
    "files.sectionTitle": "Downloadable Resources",
    "files.sectionDesc": "Technical guide, verification script and LTI stability patch.",
    "files.guide": "Technical Guide 2026",
    "files.guidePdf": "PDF — Full document",
    "files.guideDocx": "DOCX — Editable version",
    "files.quickstart": "Quickstart Script",
    "files.quickstartDesc": "Python — Minimal verification",
    "files.patch": "LTI Patch",
    "files.patchDesc": "Diff — Numerical stability",

    // Roadmap
    "road.sectionBadge": "Roadmap",
    "road.sectionTitle": "From Prototype to Testbed",
    "road.sectionDesc": "Demonstrate overfitting, stable loss and predictable behavior when increasing n_loops before scaling.",
    "road.phase0": "PHASE 0",
    "road.phase0Title": "Freeze version",
    "road.phase0Desc": "Repository fork, commit tag and requirements-lock for full reproducibility.",
    "road.phase1": "PHASE 1",
    "road.phase1Title": "Fix operational stability",
    "road.phase1Desc": "Apply LTI patch to prevent A from reaching exactly 1.0 in float32.",
    "road.phase2": "PHASE 2",
    "road.phase2Title": "Add packaging",
    "road.phase2Desc": "pyproject.toml, editable install and simple CLI.",
    "road.phase3": "PHASE 3",
    "road.phase3Title": "Tokenizer and data",
    "road.phase3Desc": "Define tokenizer, dataset and causal LM format.",
    "road.phase4": "PHASE 4",
    "road.phase4Title": "Minimal training",
    "road.phase4Desc": "train.py script with warmup, clipping, checkpoints and logging.",
    "road.phase5": "PHASE 5",
    "road.phase5Title": "Benchmarks",
    "road.phase5Desc": "Curves vs loops, GQA vs MLA comparison, MoE/LoRA/ACT ablations.",
    "road.phase6": "PHASE 6",
    "road.phase6Title": "Publication and comparison",
    "road.phase6Desc": "Comparisons against dense transformers of equal FLOPs budget.",
    "road.done": "Completed",

    // Contribute
    "contribute.sectionBadge": "Contribute",
    "contribute.sectionTitle": "Join the Lab",
    "contribute.sectionDesc": "MythosForge is an open project. There are many ways to collaborate, from experiments to documentation.",
    "contribute.issues.title": "Report Issues",
    "contribute.issues.desc": "Find bugs, documentation errors or possible improvements. Open an issue with clear context and reproduction steps.",
    "contribute.prs.title": "Submit Pull Requests",
    "contribute.prs.desc": "Code patches, architecture improvements, new experiments or integrations. Read the CONTRIBUTING.md guide first.",
    "contribute.discuss.title": "Discuss Ideas",
    "contribute.discuss.desc": "Propose experiments, share results or debate about the RDT architecture in the repository discussions.",

    // Bibliography
    "refs.sectionBadge": "Bibliography",
    "refs.sectionTitle": "Academic References",
    "refs.sectionDesc": "Key papers that support the technical hypothesis.",

    // Footer
    "footer.brand": '<span class="text-cyan">Mythos</span><span class="text-gold">Forge</span> — Created by <a href="https://github.com/smouj" target="_blank" rel="noopener" style="color:var(--cyan)">Smouj013</a> with AI agents',
    "footer.copy": "Independent technical document — April 2026",
  }
};

// ─── Language Switcher Logic ───

function getSavedLang() {
  try {
    return localStorage.getItem('mythosforge-lang') || 'es';
  } catch { return 'es'; }
}

function setLang(lang) {
  try { localStorage.setItem('mythosforge-lang', lang); } catch {}
  applyLang(lang);
  updateSwitcherUI(lang);
  document.documentElement.lang = lang;
}

function applyLang(lang) {
  const t = translations[lang];
  if (!t) return;

  // Update all [data-i18n] elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (t[key]) {
      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        el.placeholder = t[key];
      } else {
        el.innerHTML = t[key];
      }
    }
  });

  // Update page title
  if (lang === 'en') {
    document.title = 'MythosForge — Recurrent-Depth Transformer Research Lab';
  } else {
    document.title = 'MythosForge — Laboratorio de Investigación RDT';
  }
}

function updateSwitcherUI(lang) {
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  const lang = getSavedLang();
  applyLang(lang);
  updateSwitcherUI(lang);

  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      setLang(btn.dataset.lang);
    });
  });
});
