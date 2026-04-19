# Contributing to MythosForge

Gracias por tu interés en contribuir a MythosForge. Este documento describe las normas y flujos de trabajo para colaborar en el proyecto.

## Código de Conducta

- Sé respetuoso y constructivo en todas las interacciones (issues, PRs, discusiones).
- No se tolerarán comentarios ofensivos, discriminatorios ni acosadores.
- Mantén las discusiones técnicas y enfocadas en el proyecto.

## Tipos de Contribuciones

### Experimentos y Benchmarks
- Ejecutar OpenMythos con distintas configuraciones (MLA vs GQA, distintos n_loops, etc.).
- Medir latencia, uso de memoria, pérdida y calidad de generación.
- Comparar contra transformers densos de igual presupuesto de FLOPs.

### Parches y Optimizaciones
- Mejoras de estabilidad numérica (como el parche LTI existente).
- Optimizaciones de rendimiento (kernel fusion, compilación, cuantización).
- Nuevos adaptadores de atención o variantes de MoE.

### Documentación
- Correcciones y ampliaciones de la guía técnica.
- Traducciones a otros idiomas.
- Tutoriales y notebooks interactivos.
- Mejoras al README y GitHub Pages.

### Herramientas y Visualizaciones
- Scripts de diagnóstico (distribución de expertos, halting probabilities, espectro de A).
- Dashboards de entrenamiento.
- Visualizaciones comparativas.

## Flujo de Trabajo

1. **Abre un Issue** antes de trabajar en algo significativo. Describe qué quieres hacer y por qué.
2. **Fork** el repositorio y crea una rama descriptiva:
   ```bash
   git checkout -b feat/moe-ablation-study
   git checkout -b fix/mla-memory-leak
   git checkout -b docs/training-guide
   ```
3. **Desarrolla** tus cambios con commits claros y atómicos.
4. **Testea** tus cambios localmente. Si es código Python, asegúrate de que los tests existentes pasan:
   ```bash
   pytest -q
   ```
5. **Abre un Pull Request** con:
   - Descripción clara de los cambios.
   - Contexto sobre por qué son útiles.
   - Capturas de pantalla si afectan la UI.
   - Referencia al Issue relacionado (`Closes #XX`).

## Normas de Código

- **Python**: PEP 8, type hints donde sea razonable.
- **TypeScript**: Strict mode, componentes funcionales.
- **Commits**: Formato conventional (`feat:`, `fix:`, `docs:`, `refactor:`, etc.).
- **No se aceptan**: secrets, tokens, claves API ni datos sensibles en ningún archivo.

## Reportar Issues

Al abrir un Issue, incluye:

- **Descripción clara** del problema o propuesta.
- **Entorno**: OS, versión de Python, versión de PyTorch, GPU (si aplica).
- **Pasos para reproducir** (si es un bug).
- **Logs o traces** relevantes.
- **Comportamiento esperado** vs. comportamiento observado.

## Estructura del Repositorio

```
MythosForge/
├── docs/          # GitHub Pages (sitio estático)
├── src/           # Archivos de investigación (guía, scripts, parches)
├── .github/       # Templates de Issues y PRs
├── LICENSE        # MIT
├── README.md      # Documentación principal
└── CONTRIBUTING.md # Este archivo
```

## Preguntas

Si tienes dudas, abre un Issue con la etiqueta `question` o usa las [Discusiones](https://github.com/smouj/MythosForge/discussions) del repositorio.

---

*Hecho con dedication por [Smouj013](https://github.com/smouj) con agentes de IA.*
