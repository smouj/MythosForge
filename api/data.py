"""
MythosForge API — Static project data.

All data sourced from the real MythosForge/OpenMythos project.
No mocks, no placeholders — everything reflects actual implementation.
"""

from api.models import (
    ProjectInfo, ArchBlock, ArchComponent, ComponentDetail,
    ValidationCheck, CheckStatus, RepoStatusItem,
    RoadmapPhase, PhaseStatus, Reference, Link
)


# ─── Project Info ─────────────────────────────────────

PROJECT_INFO = ProjectInfo(
    name="MythosForge",
    full_name="MythosForge — Recurrent-Depth Transformer Research Lab",
    version="0.2.0",
    api_version="v1",
    description=(
        "Laboratorio de investigación sobre transformers recurrentes en profundidad, "
        "razonamiento latente, MoE, atención MLA/GQA, inyección LTI estable y halting adaptativo."
    ),
    description_en=(
        "Research lab on recurrent-depth transformers, latent reasoning, MoE, "
        "switchable MLA/GQA attention, stable LTI injection, and adaptive halting."
    ),
    license="MIT",
    repository="https://github.com/smouj/MythosForge",
    github_pages="https://smouj.github.io/MythosForge/",
    creator="Smouj013",
    created_with="AI Agents",
    date="2026-04",
    based_on="OpenMythos (https://github.com/kyegomez/OpenMythos)",
    disclaimer=(
        "OpenMythos se presenta como reconstrucción teórica independiente, "
        "no afiliada a Anthropic. MythosForge no afirma equivalencia con ningún sistema propietario."
    ),
    features=[
        "Replicar el repositorio y ejecutarlo",
        "Documentar la arquitectura completa",
        "Inferencia mínima con GQA y MLA",
        "Proporcionar parche de estabilidad LTI",
        "Servir como base de investigación seria",
        "API REST con datos reales del proyecto",
        "Endpoint de inferencia con OpenMythos",
    ],
    not_claims=[
        "Es una réplica de Claude Mythos",
        "Tiene pesos preentrenados",
        "Equivalencia con Anthropic",
        "Es un producto de Anthropic",
    ],
)


# ─── Architecture ────────────────────────────────────

ARCH_FLOW = [
    ArchBlock(name="Input Tokens", description="", execution=""),
    ArchBlock(
        name="PRELUDE",
        description="Bloque Transformer denso con FFN SwiGLU",
        description_en="Dense Transformer block with FFN SwiGLU",
        execution="1x",
    ),
    ArchBlock(
        name="RECURRENT BLOCK",
        description="TransformerBlock compartido con loop recurrente",
        description_en="Shared TransformerBlock with recurrent loop",
        execution="Nx (configurable via max_loop_iters)",
    ),
    ArchBlock(
        name="CODA",
        description="Bloque Transformer denso con FFN SwiGLU y proyección final",
        description_en="Dense Transformer block with FFN SwiGLU and final projection",
        execution="1x",
    ),
    ArchBlock(name="RMSNorm", description="Root Mean Square Layer Normalization", execution=""),
    ArchBlock(name="LM Head", description="Proyección al vocabulario", execution=""),
    ArchBlock(name="Output Logits", description="Distribución sobre el vocabulario", execution=""),
]

RECURRENT_COMPONENTS = [
    ArchComponent(
        name="MLA / GQA",
        short_name="mla-gqa",
        description="Atención conmutable: GQA reduce caché KV, MLA comprime en latente de bajo rango",
        description_en="Switchable attention: GQA reduces KV cache, MLA compresses into low-rank latent",
    ),
    ArchComponent(
        name="MoE",
        short_name="moe",
        description="Mixture of Experts: routed top-K + shared experts siempre activos (DeepSeekMoE)",
        description_en="Mixture of Experts: routed top-K + always-active shared experts (DeepSeekMoE)",
    ),
    ArchComponent(
        name="LoRA",
        short_name="lora",
        description="Adaptador Low-Rank Adaptation por iteración del loop, señal de profundidad via embedding sinusoidal",
        description_en="Low-Rank Adaptation adapter per loop iteration, depth signal via sinusoidal embedding",
    ),
    ArchComponent(
        name="LTI",
        short_name="lti",
        description="Linear Time-Invariant injection con A ∈ (0,1) por parametrización log-space",
        description_en="Linear Time-Invariant injection with A ∈ (0,1) via log-space parametrization",
    ),
    ArchComponent(
        name="ACT",
        short_name="act",
        description="Adaptive Computation Time: halting adaptativo por posición de secuencia",
        description_en="Adaptive Computation Time: adaptive halting per sequence position",
    ),
]

ARCHITECTURE = {
    "flow": ARCH_FLOW,
    "recurrent_components": RECURRENT_COMPONENTS,
    "recurrent_rule": "h_(t+1) = A · h_t + B · e + Transformer(h_t, e)",
    "recurrent_rule_desc": (
        "El razonamiento intermedio se acumula en el estado oculto h_t sin expresarse como tokens visibles. "
        "La matriz A ∈ (0,1) garantiza estabilidad espectral por construcción paramétrica."
    ),
    "recurrent_rule_desc_en": (
        "Intermediate reasoning accumulates in hidden state h_t without being expressed as visible tokens. "
        "Matrix A ∈ (0,1) guarantees spectral stability by parametric construction."
    ),
    "max_loop_iters": 32,
}


# ─── Components ───────────────────────────────────────

COMPONENTS = [
    ComponentDetail(
        slug="mla-gqa",
        name="Atención Conmutable MLA / GQA",
        name_en="Switchable MLA / GQA Attention",
        description=(
            "GQA reduce la caché KV compartiendo pares K/V entre grupos de cabezas de consulta. "
            "MLA comprime la ruta KV en un latente de bajo rango y reconstruye K/V al vuelo. "
            "Conmutable mediante configuración: attn_type='gqa' o attn_type='mla'."
        ),
        description_en=(
            "GQA reduces KV cache by sharing K/V pairs across query head groups. "
            "MLA compresses the KV route into a low-rank latent and reconstructs K/V on the fly. "
            "Switchable via configuration: attn_type='gqa' or attn_type='mla'."
        ),
        color="#00D4FF",
        references=[
            Link(url="https://arxiv.org/abs/2405.04434", label="DeepSeek-V2 (MLA)"),
            Link(url="https://arxiv.org/abs/2305.13245", label="GQA Paper"),
        ],
        key_insight="MLA consigue mayor compresión de memoria; GQA ofrece menor latencia. La elección depende del balance memoria/velocidad.",
    ),
    ComponentDetail(
        slug="moe",
        name="MoE en Bloque Recurrente",
        name_en="MoE in Recurrent Block",
        description=(
            "Routed experts top-K + shared experts siempre activos. "
            "Hereda de DeepSeekMoE la segmentación fina y el aislamiento de expertos compartidos. "
            "Maximiza la relación rendimiento/cómputo dentro del bloque recurrente."
        ),
        description_en=(
            "Routed top-K experts + always-active shared experts. "
            "Inherits fine-grained segmentation and shared expert isolation from DeepSeekMoE. "
            "Maximizes performance/compute ratio within the recurrent block."
        ),
        color="#FFB800",
        references=[
            Link(url="https://arxiv.org/abs/2401.06066", label="DeepSeekMoE"),
        ],
        key_insight="Dentro de un bloque recurrente compartido, MoE permite especialización por iteración sin aumentar parámetros estáticos.",
    ),
    ComponentDetail(
        slug="lti",
        name="Inyección LTI Estable",
        name_en="Stable LTI Injection",
        description=(
            "Linear Time-Invariance injection con parametrización log-space. "
            "Garantiza A ∈ (0,1) por construcción, manteniendo estabilidad espectral. "
            "Parche numérico incluido para evitar saturación float32 en el borde de 1.0."
        ),
        description_en=(
            "Linear Time-Invariance injection with log-space parametrization. "
            "Guarantees A ∈ (0,1) by construction, maintaining spectral stability. "
            "Numeric patch included to prevent float32 saturation at the 1.0 boundary."
        ),
        color="#00E87B",
        references=[
            Link(url="https://arxiv.org/abs/2604.12946", label="Parcae"),
        ],
        key_insight="La estabilidad espectral es fundamental: sin ella, los loops recurrentes pueden diverger tras pocas iteraciones.",
    ),
    ComponentDetail(
        slug="act",
        name="ACT — Halting Adaptativo",
        name_en="ACT — Adaptive Halting",
        description=(
            "Adaptive Computation Time aprende una probabilidad de halting por posición de secuencia. "
            "Permite que algunas posiciones dejen de acumular actualizaciones antes que otras. "
            "Optimiza el cómputo de inferencia de forma adaptativa."
        ),
        description_en=(
            "Adaptive Computation Time learns a halting probability per sequence position. "
            "Some positions stop accumulating updates before others. "
            "Optimizes inference computation adaptively."
        ),
        color="#A78BFA",
        references=[
            Link(url="https://arxiv.org/abs/1603.08983", label="ACT (Graves, 2016)"),
        ],
        key_insight="ACT es la clave para test-time compute escalable: el modelo decide cuánto razonamiento necesita por posición.",
    ),
    ComponentDetail(
        slug="lora",
        name="Adaptador LoRA por Iteración",
        name_en="Per-Iteration LoRA Adapter",
        description=(
            "Cada iteración del bucle puede modificar su comportamiento con un adaptador LoRA "
            "dependiente de la iteración. Enriquece la representación sin multiplicar parámetros estáticos. "
            "Señal de profundidad mediante embedding sinusoidal de índice de loop."
        ),
        description_en=(
            "Each loop iteration can modify its behavior with an iteration-dependent LoRA adapter. "
            "Enriches the representation without multiplying static parameters. "
            "Depth signal via sinusoidal embedding of loop index."
        ),
        color="#FF6B9D",
        references=[],
        key_insight="LoRA por iteración permite que cada paso del loop tenga comportamiento diferente con coste mínimo de parámetros.",
    ),
    ComponentDetail(
        slug="prelude-coda",
        name="Prelude + Recurrent + Coda",
        name_en="Prelude + Recurrent + Coda",
        description=(
            "Arquitectura de tres etapas: bloques densos de entrada (Prelude) y salida (Coda) "
            "ejecutados una vez, con bloque recurrente compartido aplicado N veces en el medio. "
            "Ambos bloques densos usan FFN SwiGLU."
        ),
        description_en=(
            "Three-stage architecture: input (Prelude) and output (Coda) dense blocks "
            "executed once, with a shared recurrent block applied N times in between. "
            "Both dense blocks use FFN SwiGLU."
        ),
        color="#00D4FF",
        references=[
            Link(url="https://arxiv.org/abs/2604.07822", label="Loop, Think, & Generalize"),
            Link(url="https://arxiv.org/abs/1807.03819", label="Universal Transformers"),
        ],
        key_insight="Prelude y Coda procesan el input/output una vez; toda la profundidad viene de repetir el bloque central.",
    ),
]


# ─── Validation ───────────────────────────────────────

INFERENCE_CHECKS = [
    ValidationCheck(
        name="Import y construcción del modelo",
        name_en="Model import and construction",
        status=CheckStatus.PASS,
    ),
    ValidationCheck(
        name="Forward mínimo con GQA",
        name_en="Minimal forward with GQA",
        status=CheckStatus.PASS,
    ),
    ValidationCheck(
        name="Forward mínimo con MLA",
        name_en="Minimal forward with MLA",
        status=CheckStatus.PASS,
    ),
    ValidationCheck(
        name="Generación mínima autoregresiva",
        name_en="Minimal autoregressive generation",
        status=CheckStatus.PASS,
    ),
    ValidationCheck(
        name="Matriz A de LTI en rango estable",
        name_en="LTI A matrix in stable range",
        status=CheckStatus.PASS,
    ),
    ValidationCheck(
        name="test_spectral_radius_stable_after_large_grad_step",
        name_en="test_spectral_radius_stable_after_large_grad_step",
        status=CheckStatus.WARN,
        note="Fallo numérico float32 en borde 1.0 — parche disponible",
        note_en="float32 numeric failure at 1.0 boundary — patch available",
    ),
]

REPO_STATUS = [
    RepoStatusItem(feature="Repositorio público", feature_en="Public repository", available=True),
    RepoStatusItem(feature="Licencia MIT", feature_en="MIT License", available=True),
    RepoStatusItem(feature="Implementación PyTorch", feature_en="PyTorch implementation", available=True),
    RepoStatusItem(feature="Ejemplo mínimo de uso", feature_en="Minimal usage example", available=True),
    RepoStatusItem(feature="Pruebas automatizadas", feature_en="Automated tests", available=True),
    RepoStatusItem(feature="Pesos preentrenados", feature_en="Pretrained weights", available=False),
    RepoStatusItem(feature="Tokenizer propio", feature_en="Custom tokenizer", available=False),
    RepoStatusItem(feature="Pipeline entrenamiento completo", feature_en="Complete training pipeline", available=False),
    RepoStatusItem(feature="Equivalencia Claude Mythos", feature_en="Claude Mythos equivalence", available=False),
]


# ─── Roadmap ──────────────────────────────────────────

ROADMAP = [
    RoadmapPhase(phase=0, name="Congelar versión", name_en="Freeze version",
                 description="Fork del repositorio, etiqueta de commit y requirements-lock para reproducibilidad total.",
                 description_en="Repository fork, commit tag and requirements-lock for full reproducibility.",
                 status=PhaseStatus.COMPLETED),
    RoadmapPhase(phase=1, name="Corregir estabilidad operativa", name_en="Fix operational stability",
                 description="Aplicar parche LTI para evitar que A alcance exactamente 1.0 en float32.",
                 description_en="Apply LTI patch to prevent A from reaching exactly 1.0 in float32.",
                 status=PhaseStatus.COMPLETED),
    RoadmapPhase(phase=2, name="Añadir empaquetado", name_en="Add packaging",
                 description="pyproject.toml, instalación editable y CLI simple.",
                 description_en="pyproject.toml, editable install and simple CLI.",
                 status=PhaseStatus.PENDING),
    RoadmapPhase(phase=3, name="Tokenizer y datos", name_en="Tokenizer and data",
                 description="Definir tokenizer, dataset y formato causal LM.",
                 description_en="Define tokenizer, dataset and causal LM format.",
                 status=PhaseStatus.PENDING),
    RoadmapPhase(phase=4, name="Entrenamiento mínimo", name_en="Minimal training",
                 description="Script train.py con warmup, clipping, checkpoints y logging.",
                 description_en="train.py script with warmup, clipping, checkpoints and logging.",
                 status=PhaseStatus.PENDING),
    RoadmapPhase(phase=5, name="Benchmarks", name_en="Benchmarks",
                 description="Curvas vs loops, comparación GQA vs MLA, ablations MoE/LoRA/ACT.",
                 description_en="Curves vs loops, GQA vs MLA comparison, MoE/LoRA/ACT ablations.",
                 status=PhaseStatus.PENDING),
    RoadmapPhase(phase=6, name="Publicación y comparativa", name_en="Publication and comparison",
                 description="Comparativas contra transformers densos de igual presupuesto de FLOPs.",
                 description_en="Comparisons against dense transformers of equal FLOPs budget.",
                 status=PhaseStatus.PENDING),
]


# ─── References ───────────────────────────────────────

REFERENCES = [
    Reference(id="R1", title="OpenMythos README", url="https://github.com/kyegomez/OpenMythos", type="code"),
    Reference(id="R2", title="OpenMythos docs/open_mythos.md", url="https://github.com/kyegomez/OpenMythos/blob/main/docs/open_mythos.md", type="docs"),
    Reference(id="R3", title="Thread — Kye Gomez", url="https://threadreaderapp.com/user/KyeGomezB", type="thread"),
    Reference(id="R4", title="Loop, Think, & Generalize: Implicit Reasoning in RDT", url="https://arxiv.org/abs/2604.07822", type="paper"),
    Reference(id="R5", title="Parcae: Scaling Laws for Stable Looped LMs", url="https://arxiv.org/abs/2604.12946", type="paper"),
    Reference(id="R6", title="Reasoning with Latent Thoughts: Looped Transformers", url="https://arxiv.org/abs/2502.17416", type="paper"),
    Reference(id="R7", title="Coconut: Continuous Latent Space Reasoning", url="https://arxiv.org/abs/2412.06769", type="paper"),
    Reference(id="R8", title="DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model (MLA)", url="https://arxiv.org/abs/2405.04434", type="paper"),
    Reference(id="R9", title="GQA: Training Generalized Multi-Query Transformer Models", url="https://arxiv.org/abs/2305.13245", type="paper"),
    Reference(id="R10", title="DeepSeekMoE: Towards Ultimate Expert Specialization", url="https://arxiv.org/abs/2401.06066", type="paper"),
    Reference(id="R11", title="Universal Transformers", url="https://arxiv.org/abs/1807.03819", type="paper"),
    Reference(id="R12", title="Adaptive Computation Time for Recurrent Neural Networks", url="https://arxiv.org/abs/1603.08983", type="paper"),
]


# ─── i18n ─────────────────────────────────────────────

I18N_TRANSLATIONS = {
    "es": {
        "nav.architecture": "Arquitectura",
        "nav.components": "Componentes",
        "nav.validation": "Validación",
        "nav.guide": "Guía",
        "nav.files": "Archivos",
        "nav.roadmap": "Ruta",
        "nav.refs": "Bibliografía",
        "nav.contribute": "Contribuir",
        "hero.badge1": "Z.AI Research Lab",
        "hero.badge2": "Open Source",
        "hero.title1": "Mythos",
        "hero.title2": "Forge",
        "hero.desc": "Laboratorio de investigación sobre Recurrent-Depth Transformers: razonamiento latente, MoE, atención MLA/GQA y halting adaptativo.",
        "hero.btn1": "Explorar Arquitectura",
        "hero.btn2": "Guía Rápida",
        "hero.disclaimer": "OpenMythos se presenta como reconstrucción teórica independiente, no afiliada a Anthropic. Este laboratorio no afirma equivalencia con ningún sistema propietario.",
        "arch.sectionBadge": "Arquitectura",
        "arch.sectionTitle": "Flujo Operativo",
        "arch.sectionDesc": "La profundidad útil no viene de apilar capas distintas, sino de aplicar varias veces el mismo bloque compartido durante un único forward pass.",
        "comp.sectionBadge": "Componentes",
        "comp.sectionTitle": "Piezas Arquitectónicas",
        "comp.sectionDesc": "Cada componente seleccionado por coherencia con la literatura más reciente en transformers recurrentes y razonamiento latente.",
        "val.sectionBadge": "Validación",
        "val.sectionTitle": "Pruebas Ejecutadas",
        "val.sectionDesc": "Validación práctica en CPU con PyTorch 2.10.0+cpu. Configuraciones pequeñas verificadas.",
        "road.sectionBadge": "Hoja de Ruta",
        "road.sectionTitle": "De Prototipo a Banco de Pruebas",
        "road.sectionDesc": "Demostrar overfitting, pérdida estable y comportamiento predecible al aumentar n_loops antes de escalar.",
        "road.done": "Completada",
        "refs.sectionBadge": "Bibliografía",
        "refs.sectionTitle": "Referencias Académicas",
        "refs.sectionDesc": "Trabajos clave que sostienen la hipótesis técnica.",
        "footer.brand": "MythosForge — Creado por Smouj013 con agentes de IA",
        "footer.copy": "Documento técnico independiente — Abril 2026",
        "contribute.sectionBadge": "Contribuir",
        "contribute.sectionTitle": "Únete al Laboratorio",
        "contribute.sectionDesc": "MythosForge es un proyecto abierto. Hay muchas formas de colaborar, desde experimentos hasta documentación.",
    },
    "en": {
        "nav.architecture": "Architecture",
        "nav.components": "Components",
        "nav.validation": "Validation",
        "nav.guide": "Guide",
        "nav.files": "Files",
        "nav.roadmap": "Roadmap",
        "nav.refs": "Bibliography",
        "nav.contribute": "Contribute",
        "hero.badge1": "Z.AI Research Lab",
        "hero.badge2": "Open Source",
        "hero.title1": "Mythos",
        "hero.title2": "Forge",
        "hero.desc": "Research lab on Recurrent-Depth Transformers: latent reasoning, MoE, switchable MLA/GQA attention and adaptive halting.",
        "hero.btn1": "Explore Architecture",
        "hero.btn2": "Quickstart Guide",
        "hero.disclaimer": "OpenMythos is presented as an independent theoretical reconstruction, not affiliated with Anthropic. This lab does not claim equivalence with any proprietary system.",
        "arch.sectionBadge": "Architecture",
        "arch.sectionTitle": "Operational Flow",
        "arch.sectionDesc": "Useful depth doesn't come from stacking different layers — it comes from applying the same shared block multiple times during a single forward pass.",
        "comp.sectionBadge": "Components",
        "comp.sectionTitle": "Architectural Pieces",
        "comp.sectionDesc": "Each component selected for coherence with the latest literature on recurrent transformers and latent reasoning.",
        "val.sectionBadge": "Validation",
        "val.sectionTitle": "Executed Tests",
        "val.sectionDesc": "Practical validation on CPU with PyTorch 2.10.0+cpu. Small configurations verified.",
        "road.sectionBadge": "Roadmap",
        "road.sectionTitle": "From Prototype to Testbed",
        "road.sectionDesc": "Demonstrate overfitting, stable loss and predictable behavior when increasing n_loops before scaling.",
        "road.done": "Completed",
        "refs.sectionBadge": "Bibliography",
        "refs.sectionTitle": "Academic References",
        "refs.sectionDesc": "Key papers that support the technical hypothesis.",
        "footer.brand": "MythosForge — Created by Smouj013 with AI agents",
        "footer.copy": "Independent technical document — April 2026",
        "contribute.sectionBadge": "Contribute",
        "contribute.sectionTitle": "Join the Lab",
        "contribute.sectionDesc": "MythosForge is an open project. There are many ways to collaborate, from experiments to documentation.",
    },
}
