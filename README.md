# RAG-VECTORIAL

Implementación de RAG para las FAQs de Parachute S.A. con PostgreSQL, pgvector y
function calling.

## Video demostrativo

[![Previsualización del video demostrativo](https://img.youtube.com/vi/0NCuBv0_-zY/hqdefault.jpg)](https://youtu.be/0NCuBv0_-zY)

Haz clic en la imagen para ver la demostración del cargador y del agente.

## Entorno virtual

Desde la raíz del repositorio, crea y activa el entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

En PowerShell de Windows, actívalo con:

```powershell
.venv\Scripts\Activate.ps1
```

Con el entorno activo, instala todas las dependencias del proyecto una sola vez:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`main.py` es el punto de entrada del proyecto: usa `load` para cargar el corpus y
`chat` para iniciar el agente.

## Infraestructura

La base vectorial se ejecuta localmente con PostgreSQL 16 y la extensión pgvector
mediante Docker Compose. Los datos se guardan en el volumen `postgres_data`, por
lo que permanecen disponibles aunque el contenedor se detenga.

1. Instala Docker Desktop o Docker Engine con el complemento Docker Compose.
2. Desde la raíz del repositorio, inicia la base de datos:

   ```bash
   docker compose up -d
   ```

3. Comprueba que PostgreSQL ya acepta conexiones:

   ```bash
   docker compose ps
   ```

   El estado debe mostrarse como `healthy`.

4. Para detener la infraestructura conservando sus datos:

   ```bash
   docker compose down
   ```

Antes de iniciarla, define en `.env` las variables `POSTGRES_DB`, `POSTGRES_USER`,
`POSTGRES_PASSWORD` y `DATABASE_URL`. La URL debe seguir el formato
`postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@localhost:5432/<POSTGRES_DB>`.
El archivo `.env` no se versiona; `.env.example` solo documenta las variables
requeridas y no contiene valores de configuración.

> Para reiniciar la base desde cero, use `docker compose down -v`. Esto elimina
> permanentemente el volumen con los embeddings y los datos cargados.

### Esquema de FAQs

Una vez iniciada la base, aplica el esquema de forma explícita:

```bash
docker compose exec -T postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < db/schema.sql
```

El esquema crea la extensión `vector`, la tabla `faqs` y un índice HNSW para
búsquedas por distancia coseno. La columna `embedding` tiene tipo `vector(384)`
porque el proyecto usa el modelo multilingüe
configurado en `EMBEDDING_MODEL`, que genera vectores de esa dimensión.

## Carga del corpus

`load_faqs.py` lee el corpus, valida su formato, crea un embedding normalizado por
pregunta y realiza un *upsert* en PostgreSQL. Por ello se puede ejecutar nuevamente
después de cambiar el corpus sin duplicar registros.

Con el entorno virtual activo, ejecuta el cargador desde la raíz del repositorio:

```bash
python main.py load
```

El modelo se descarga en la primera ejecución. El cargador requiere `DATABASE_URL`,
`FAQ_TABLE` y `EMBEDDING_MODEL` en `.env`; aplica `db/schema.sql` antes de insertar
las FAQs, por lo que también funciona con una base nueva.

Para cambiar el tamaño de lote, use `python main.py load --batch-size 16`.

## Agente de IA

`agent.py` es el programa conversacional. No ingiere el corpus: consulta la tabla
vectorial que genere el script de carga.

Copia `.env.example` como `.env` y configura las credenciales y la conexión de la
base de datos. Se admite `OPENAI_API_KEY` o `NVIDIA_API_KEY`; `OPENAI_BASE_URL` y
`MODEL` permiten utilizar un endpoint compatible con el SDK de OpenAI, como el
configurado en el proyecto anterior.

Después de que la infraestructura y la carga estén listas, ejecuta:

```powershell
python main.py chat
```

La sesión acepta múltiples preguntas y termina al escribir `Bye` o al usar
`Ctrl-C`.

### Contrato con el script de carga

Para integrarse con `agent.py`, la tabla (por defecto `faqs`) debe tener estas
columnas. El agente genera la consulta con
`paraphrase-multilingual-MiniLM-L12-v2`, normalizada, por lo
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
cinco FAQs. Los resultados por debajo de `MINIMUM_SIMILARITY` se descartan para que
el agente admita cuando no dispone de información suficiente. Si el cargador usa
otro nombre de tabla, configúralo con `FAQ_TABLE`. El valor de
`MINIMUM_SIMILARITY` se define en `.env` y debe estar entre `0` y `1`.
