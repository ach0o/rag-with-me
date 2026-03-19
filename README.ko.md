[English](README.md)

# RAG Agent on LangGraph

Python과 LangGraph로 만든 모듈형 RAG(Retrieval-Augmented Generation) 에이전트입니다. 청커, 벡터 스토어, 임베더, LLM, 리트리버, 리랭커 등 모든 컴포넌트가 추상화 뒤에 있어서 YAML 설정만 바꾸면 자유롭게 교체할 수 있습니다.

주요 데이터 소스는 **97 Things Every Programmer Should Know** 마크다운 파일 모음입니다.

## 아키텍처

**헥사고날 아키텍처(포트 & 어댑터)** 구조에, 데이터 수집은 **파이프라인 패턴**으로 처리합니다.

```
src/rag_agent/
├── domain/           # 순수 비즈니스 모델과 포트 (Protocol 클래스). 외부 의존성 없음.
│   ├── models.py     # Document, Chunk, QueryResult
│   ├── pipeline.py   # Pipeline, PipelineStage
│   └── ports.py      # 모든 포트 정의
├── application/      # 유스케이스 오케스트레이션. domain/만 import.
│   ├── ingest.py     # IngestUseCase + 파이프라인 스테이지
│   ├── query.py      # QueryUseCase (단순 선형 흐름)
│   └── query_graph.py # QueryGraphBuilder (LangGraph 상태 머신)
├── adapters/
│   ├── inbound/      # CLI 인자 파싱
│   └── outbound/     # 포트의 구현체들
│       ├── chunkers/       # fixed-size, markdown-header, semantic
│       ├── doc_loaders/    # markdown
│       ├── embedders/      # Azure OpenAI
│       ├── llms/           # Azure OpenAI
│       ├── vector_stores/  # ChromaDB
│       ├── retrievers/     # dense, BM25 sparse, hybrid (RRF)
│       ├── rerankers/      # none, Cohere, cross-encoder
│       ├── document_repos/ # PostgreSQL
│       └── chunk_repos/    # PostgreSQL
├── config.py         # Pydantic 모델, YAML 로더
└── main.py           # 컴포지션 루트 (어댑터 → 유스케이스 연결)
```

### 의존성 규칙

- `domain/`은 외부 라이브러리를 일절 import하지 않습니다 — 순수 Python만 사용
- `application/`은 `domain/`만 import합니다
- `adapters/`는 `domain/`과 외부 라이브러리를 import합니다
- `main.py`만 `adapters/`를 import할 수 있습니다

## 교체 가능한 컴포넌트

| 슬롯 | 사용 가능한 어댑터 | 설정 키 |
|------|-------------------|---------|
| LLM | Azure OpenAI | `llm.provider` |
| 임베더 | Azure OpenAI | `embedder.provider` |
| 벡터 스토어 | ChromaDB | `vector_store.provider` |
| 문서 로더 | markdown, pdf | `data_source.types` |
| 청커 | fixed-size, markdown-header, semantic | `chunker.strategy` |
| 리트리버 | dense, bm25_sparse, hybrid | `retriever.provider` |
| 리랭커 | none, cohere, cross_encoder | `reranker.provider` |
| 문서 저장소 | PostgreSQL | `database.enabled` |
| 청크 저장소 | PostgreSQL | `database.enabled` |

`config/default.yaml`만 수정하면 됩니다 — 코드 변경 없이 컴포넌트를 교체할 수 있습니다.

## 데이터 흐름

### 수집 (Ingestion)

```
CLI → DocLoader.load() (경로 × 타입별) → [문서 저장] → Chunker.chunk()
    → [청크 저장] → Embedder.embed() → VectorStore.add()
```

여러 경로와 파일 타입을 한 번에 수집할 수 있습니다. 저장 단계는 선택사항입니다 — `database.enabled: true`일 때만 동작합니다.

### 질의 (LangGraph)

```
검색 → 컨텍스트 평가 →[충분]→ 답변 생성 → 완료
            ↓
          [부족]
            ↓
        질문 재구성 → 검색 (최대 2회 재시도)
```

평가 LLM이 검색된 컨텍스트가 질문에 답하기 충분한지 판단합니다. 부족하면 질문을 재구성하고 다시 검색합니다 (최대 2회). 단순 선형 질의 경로(`QueryUseCase`)도 폴백으로 사용 가능합니다.

## 설치 방법

### 사전 요구사항

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker (PostgreSQL용)

### 설치

```bash
uv sync
```

### 환경 변수

`.env-example`을 `.env`로 복사하고 키를 입력하세요:

```bash
cp .env-example .env
```

필수:
- `AZURE_OPENAI_API_KEY` — Azure OpenAI API 키
- `AZURE_OPENAI_EMBEDDING_ENDPOINT` — Azure OpenAI 임베딩 엔드포인트

선택 (PostgreSQL 사용 시):
- `POSTGRES_PASSWORD` — Docker PostgreSQL 비밀번호

선택 (Cohere 리랭커 사용 시):
- `COHERE_API_KEY` — Cohere API 키
- `COHERE_ENDPOINT` — Cohere rerank 엔드포인트

### 데이터베이스 (선택)

```bash
docker compose up -d
```

`config/default.yaml`에서 `database.enabled: true`로 설정하세요.

### 데이터 소스

97 Things 저장소를 `data/` 디렉토리에 클론하세요:

```bash
git clone https://github.com/97-things/97-things-every-programmer-should-know data/97-things
```

## 사용법

### 문서 수집

```bash
python -m rag_agent ingest
```

### 질의

```bash
python -m rag_agent query "코드 리뷰에 대해 프로그래머가 알아야 할 것은?"
```

### 커스텀 설정 사용

```bash
python -m rag_agent --config config/custom.yaml query "질문 내용"
```

## 설정

모든 설정은 `config/default.yaml`에 있습니다. 예시:

```yaml
llm:
  provider: azure-openai
  model: gpt-5-mini
  temperature: 0.0
  max_tokens: 1024

chunker:
  strategy: semantic        # fixed-size | markdown-header | semantic
  threshold: 0.5
  min_chunk_size: 100

retriever:
  provider: hybrid          # dense | bm25_sparse | hybrid
  top_k: 5

reranker:
  provider: cross_encoder   # none | cohere | cross_encoder
  top_k: 3

database:
  url: "postgresql://rag:localdev@localhost:5432/rag_agent"
  enabled: true
```

Pydantic이 시작 시 모든 설정을 검증합니다 — 오타나 잘못된 값은 실행 전에 바로 잡아냅니다.

## 기술 스택

| 레이어 | 선택 |
|--------|------|
| 언어 | Python 3.12 |
| 에이전트 프레임워크 | LangGraph |
| 패키지 매니저 | uv |
| 설정 | YAML + Pydantic |
| 벡터 스토어 | ChromaDB |
| 데이터베이스 | PostgreSQL 16 |
| LLM / 임베더 | Azure OpenAI |
| 테스트 | pytest + pytest-cov |
| 인프라 | Docker Compose |

## 로드맵

- [x] Phase 1 — 기반 구축 (엔드투엔드 RAG 파이프라인)
- [x] Phase 2 — 다양한 어댑터, PostgreSQL 저장소, 리랭커, 유닛 테스트
- [x] Phase 3 — LangGraph 에이전트 (평가 + 재질의 루프)
- [ ] Phase 4 — PDF/DOCX 로더, 쿼리 확장 (PDF 로더 완료, 다중 경로 수집 완료)
- [ ] Phase 5 — 평가 및 관측성
- [ ] Phase 6 — 포트폴리오 마무리 (데모 UI, CI)
