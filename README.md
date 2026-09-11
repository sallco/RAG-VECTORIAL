# RAG-VECTORIAL

Implementación de RAG para las FAQs de Parachute S.A. con PostgreSQL, pgvector y
function calling.

## Agente de IA

`agent.py` es el programa conversacional. No ingiere el corpus: consulta la tabla
vectorial que genere el script de carga. Instala sus dependencias con:

```powershell
python -m pip install -r requirements-agent.txt
```

Copia `.env.example` como `.env` y configura las credenciales y la conexión de la
base de datos. Se admite `OPENAI_API_KEY` o `NVIDIA_API_KEY`; `OPENAI_BASE_URL` y
`MODEL` permiten utilizar un endpoint compatible con el SDK de OpenAI, como el
configurado en el proyecto anterior.

Después de que la infraestructura y la carga estén listas, ejecuta:

```powershell
python agent.py
```

### Contrato con el script de carga

Para integrarse con `agent.py`, la tabla (por defecto `faqs`) debe tener estas
columnas. El agente genera la consulta con `all-MiniLM-L6-v2`, normalizada, por lo
que `embedding` debe ser `vector(384)` y usar el mismo modelo/configuración.

| Columna | Tipo esperado |
| --- | --- |
| `id` | `text` o `varchar` |
| `categoria` | `text` |
| `pregunta` | `text` |
| `respuesta` | `text` |
| `metadata` | `jsonb` |
| `embedding` | `vector(384)` |

La búsqueda utiliza distancia coseno (`embedding <=> consulta`) y devuelve hasta
cinco FAQs. Si el cargador usa otro nombre de tabla, configúralo con `FAQ_TABLE`.
