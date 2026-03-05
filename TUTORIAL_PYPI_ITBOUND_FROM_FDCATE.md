# itbound PyPI 출시 초상세 튜토리얼 (FD-CATE 성공 패턴 복제판)

이 문서는 `FD-CATE`에서 실제로 `pip install fd-cate`를 가능하게 만든 절차를 그대로 추출해, `Information-Theretic-Bounds`(로컬 폴더명: `fBound`)를 `pip install itbound` 가능 상태로 만드는 **Codex 실행용 Runbook**이다.

목표는 "설명"이 아니라 **바로 실행 가능한 운영 절차**다.

---

## 0) 최종 목표 (Definition of Done)

아래가 모두 만족되면 완료다.

1. `pip install itbound`가 새 가상환경에서 성공한다.
2. `itbound --help`가 즉시 동작한다.
3. `itbound demo` 또는 `itbound quick`가 최소 smoke 입력에서 성공한다.
4. GitHub Actions Release가 `TestPyPI -> install smoke -> PyPI` 게이트 순서로 동작한다.
5. tag 기반 릴리스(`vX.Y.Z`) 후 PyPI에서 해당 버전이 조회된다.

---

## 1) 현재 상태 스냅샷 (2026-03-05 기준)

`fBound` 저장소를 기준으로 확인된 상태:

- 이미 좋은 점
  - `pyproject.toml` 존재
  - `[project].name = "itbound"` 설정됨
  - `[project.scripts] itbound = "itbound.cli:main"` 설정됨
  - 테스트 코드(`tests/`) 다수 존재
- 아직 필요한 점
  - `.github/workflows/ci.yml` 없음
  - `.github/workflows/release.yml` 없음
  - `scripts/release_preflight.sh` 없음
  - `RELEASE_RUNBOOK.md` 없음

즉, 패키징 기본 골격은 있고, **배포 자동화/게이트 운영**이 빠진 상태다.

---

## 2) FD-CATE에서 배운 핵심 원칙 (반드시 복제)

`FD-CATE`에서 실제 성공한 패턴:

1. **PEP 621 + console script**를 먼저 고정
2. **PR CI fast gate**를 먼저 고정 (`pytest -m "not slow"` + wheel smoke)
3. **Release workflow를 4단계 게이트로 고정**
   - `build-dist`
   - `publish-testpypi`
   - `install-smoke-testpypi`
   - `publish-pypi`
4. **로컬 preflight 스크립트**를 운영해 태그 전 실수 차단
5. **PyPI 페이지 수정은 버전 단위**라는 점을 명확히 인지
   - 이미 올라간 버전의 README 렌더링은 수정 불가
   - 문제 있으면 패치 버전(`0.1.1`, `0.1.2`)로 재배포

---

## 3) 사전 준비 (한 번만)

### 3-1. TestPyPI/PyPI 계정

둘 다 준비:

- https://test.pypi.org/
- https://pypi.org/

### 3-2. 2FA 활성화 (중요)

TestPyPI/PyPI 모두 2FA 미설정이면 Trusted Publisher 설정 화면 진입이 막힌다.

### 3-3. Trusted Publisher 등록 (둘 다 동일)

프로젝트별로 다음 값 등록:

- Owner: `<GitHub owner>`
- Repository: `<repo name>`
- Workflow: `.github/workflows/release.yml`
- Environment:
  - TestPyPI: `testpypi`
  - PyPI: `pypi`
- Ref: `refs/tags/*`

---

## 4) 저장소 파일 레이아웃 목표

최소로 아래 파일을 갖춘다.

```text
fBound/
  pyproject.toml
  README.md
  CHANGELOG.md
  scripts/
    release_preflight.sh
  .github/
    workflows/
      ci.yml
      release.yml
  RELEASE_RUNBOOK.md
```

---

## 5) pyproject.toml 점검/수정 체크리스트

현재 `fBound/pyproject.toml`은 핵심 항목이 이미 좋다. 아래만 점검:

- `name = "itbound"`
- 버전 정책 (`version = "0.1.0"` 등)
- `readme = "README.md"`
- `requires-python = ">=3.9"`
- `project.scripts.itbound = "itbound.cli:main"`
- `optional-dependencies.dev`에 최소 포함
  - `pytest`
  - `build`
  - `twine`

권장 예시 (dev extras가 없다면 추가):

```toml
[project.optional-dependencies]
experiments = [
  "matplotlib",
  "scipy",
  "statsmodels",
]
dev = [
  "pytest",
  "build",
  "twine",
]
```

---

## 6) CI 워크플로 추가 (`.github/workflows/ci.yml`)

FD-CATE 패턴을 itbound에 맞게 복제한다.

핵심:

- Python matrix: `3.9`, `3.11`
- `pytest -m "not slow"`
- `python -m build`
- wheel 설치 smoke
- `itbound` CLI smoke

권장 템플릿:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  test-build:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.11"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install deps
        run: |
          python -m pip install -U pip
          python -m pip install -e .[dev]

      - name: Run fast tests
        run: |
          python -m pytest -q -m "not slow"

      - name: Build wheel/sdist
        run: |
          python -m build

      - name: Wheel smoke
        run: |
          python -m pip install dist/*.whl
          itbound --help

      - name: CLI smoke (example -> quick)
        run: |
          mkdir -p /tmp/itbound-ci
          itbound example --out /tmp/itbound-ci/example.csv
          itbound quick \
            --data /tmp/itbound-ci/example.csv \
            --treatment a \
            --outcome y \
            --covariates x1,x2 \
            --outdir /tmp/itbound-ci/out
          test -f /tmp/itbound-ci/out/results.json
```

주의:

- smoke는 "빠르고 안정적"이어야 한다.
- 큰 실험/플롯 재현은 CI fast gate에서 제외하고 `slow` marker로 분리.

---

## 7) Release 워크플로 추가 (`.github/workflows/release.yml`)

FD-CATE의 성공 파이프라인을 거의 그대로 적용한다.

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  build-dist:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Build distribution
        run: |
          python -m pip install -U pip build
          python -m build
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/*

  publish-testpypi:
    needs: build-dist
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    environment:
      name: testpypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  install-smoke-testpypi:
    needs: publish-testpypi
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install from TestPyPI and smoke
        run: |
          python -m pip install -U pip
          VERSION="${GITHUB_REF_NAME#v}"
          for i in 1 2 3 4 5; do
            python -m pip install \
              --index-url https://test.pypi.org/simple/ \
              --extra-index-url https://pypi.org/simple \
              "itbound==${VERSION}" && break
            echo "Waiting for TestPyPI propagation... attempt ${i}/5"
            sleep 20
          done
          itbound --help
          mkdir -p /tmp/itbound-testpypi
          itbound example --out /tmp/itbound-testpypi/example.csv
          itbound quick \
            --data /tmp/itbound-testpypi/example.csv \
            --treatment a \
            --outcome y \
            --covariates x1,x2 \
            --outdir /tmp/itbound-testpypi/out
          test -f /tmp/itbound-testpypi/out/results.json

  publish-pypi:
    needs: install-smoke-testpypi
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    environment:
      name: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
```

중요:

- `publish-pypi`는 반드시 `install-smoke-testpypi` 성공에 종속.
- 이 게이트가 FD-CATE에서 실제 배포 사고를 줄였다.

---

## 8) 로컬 preflight 스크립트 추가 (`scripts/release_preflight.sh`)

태그 전 반복 검증을 자동화한다.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[preflight] pytest fast"
python3 -m pytest -q -m "not slow"

echo "[preflight] build"
python3 -m build

echo "[preflight] fresh venv wheel smoke"
VENV=/tmp/itbound-release-preflight
OUT=/tmp/itbound-release-preflight-out
rm -rf "$VENV" "$OUT"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip
python -m pip install dist/*.whl
itbound --help
itbound example --out /tmp/itbound-release-preflight-example.csv
itbound quick \
  --data /tmp/itbound-release-preflight-example.csv \
  --treatment a \
  --outcome y \
  --covariates x1,x2 \
  --outdir "$OUT"
test -f "$OUT/results.json"

echo "[preflight] OK"
```

권한:

```bash
chmod +x scripts/release_preflight.sh
```

---

## 9) RELEASE_RUNBOOK.md 작성

아래 6단계는 최소 포함한다.

1. Release lane 청결 확인 (`git status` clean)
2. `scripts/release_preflight.sh` 통과
3. `pyproject.toml` 버전/`CHANGELOG.md` 동결
4. `git tag -a vX.Y.Z` + push
5. Actions에서 4단계 게이트 성공 확인
6. 새 venv에서 `pip install itbound` + smoke 실행

---

## 10) 실제 릴리스 명령 (운영 순서)

### 10-1. 로컬 검증

```bash
cd /Users/yonghanjung/Dropbox/Personal/Research/Code/fBound
bash scripts/release_preflight.sh
```

### 10-2. 버전/체인지로그 확정

- `pyproject.toml`의 `version` 갱신
- `CHANGELOG.md`에 해당 버전 항목 추가

### 10-3. 커밋/푸시

```bash
git add -A
git commit -m "release: bump version to 0.x.y"
git push origin main
```

### 10-4. 태그 푸시

```bash
git tag -a v0.x.y -m "v0.x.y"
git push origin v0.x.y
```

### 10-5. 릴리스 파이프라인 모니터링

```bash
gh run list -R yonghanjung/Information-Theretic-Bounds --workflow Release --limit 5
gh run watch <RUN_ID> -R yonghanjung/Information-Theretic-Bounds --exit-status
```

### 10-6. 최종 사용자 스모크

```bash
python3 -m venv /tmp/itbound-pypi-verify
source /tmp/itbound-pypi-verify/bin/activate
python -m pip install -U pip
python -m pip install itbound==0.x.y
itbound --help
itbound example --out /tmp/itbound-pypi-example.csv
itbound quick --data /tmp/itbound-pypi-example.csv --treatment a --outcome y --covariates x1,x2 --outdir /tmp/itbound-pypi-out
```

---

## 11) 실패 시 복구 전략

### 케이스 A: TestPyPI publish 실패

- 원인: Trusted Publisher/2FA/metadata
- 조치: 설정 수정 후 버전 증가(`0.x.(y+1)`)로 재태그

### 케이스 B: TestPyPI install smoke 실패

- 원인: 설치 후 CLI import/runtime 에러
- 조치: PyPI publish는 자동 차단되므로, 코드 수정 후 패치 버전 재태그

### 케이스 C: PyPI publish 성공 후 사용자 smoke 실패

- 조치: 즉시 `0.x.(y+1)` 패치 릴리스
- 원칙: 이미 올라간 버전은 수정 불가, 반드시 새 버전

---

## 12) Codex에게 그대로 붙여넣는 실행 프롬프트

아래를 다음 작업 턴에 그대로 사용하면 된다.

```text
목표: Information-Theretic-Bounds(fBound)를 pip install itbound 가능한 상태로 출시.

반드시 FD-CATE release 패턴을 복제해.

요구사항:
1) .github/workflows/ci.yml 추가
   - python 3.9/3.11 matrix
   - pytest -m "not slow"
   - python -m build
   - wheel install smoke
   - itbound example -> itbound quick CLI smoke

2) .github/workflows/release.yml 추가
   - build-dist -> publish-testpypi -> install-smoke-testpypi -> publish-pypi
   - publish-pypi는 install-smoke-testpypi 성공 종속
   - testpypi/pypi environment + id-token trusted publishing

3) scripts/release_preflight.sh 추가
   - fast tests + build + fresh venv wheel smoke + artifacts check

4) RELEASE_RUNBOOK.md 추가
   - 태그 전/후 체크리스트와 실패 복구 전략 포함

5) pyproject.toml / CHANGELOG.md 릴리스 버전 정합성 점검

6) 로컬에서 pytest/build/smoke 실행 결과를 남기고,
   변경 파일/명령/검증결과를 요약해.
```

---

## 13) FD-CATE에서 실제로 걸렸던 함정 (itbound도 동일 주의)

1. README 상대경로 이미지는 PyPI에서 깨진다.
- 해결: 절대 URL(`raw.githubusercontent.com/...`) 사용

2. 이미 배포된 버전 설명은 수정 불가
- 해결: 패치 버전 재배포

3. 릴리스는 "성공"이어도 사용자 설치 smoke가 마지막 진실
- 해결: fresh venv에서 실제 설치/실행을 항상 최종 게이트로 둔다

4. 테스트는 많아도 배포 실패할 수 있다
- 해결: CI green + release gate + post-release smoke를 모두 통과해야 완료로 본다

---

## 14) 바로 다음 실행 우선순위 (itbound)

1. `ci.yml` 추가
2. `release.yml` 추가
3. `release_preflight.sh` + `RELEASE_RUNBOOK.md` 추가
4. 로컬 preflight 통과
5. TestPyPI trusted publisher 확인
6. patch 버전 태그로 테스트 릴리스

이 순서가 가장 빠르고 안전하다.
