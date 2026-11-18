# lab/chefe — orquestração do lab a partir do PC chefe (Windows)

Roda no PC chefe; os PCs do lab (Ubuntu, RTX 2060 SUPER) são acessados por SSH com chave.

| Arquivo | Para quê |
|---|---|
| `dashboard.py` + `index.html` | Painel em http://127.0.0.1:8765 (GPU, VRAM, etapa, tok/s ao vivo, tabela de resultados). Lê `hosts.json` a cada 10 s. |
| `iniciar_dashboard.cmd` | Sobe o painel e abre o navegador. |
| `hosts.json` (não versionado) | `{"9.02": "<ip>", ...}`. Modelo: `hosts.example.json`. A cópia real fica no Drive (`lab/chefe/hosts.json`). |
| `lab_status.py` | Vai para `~/dory-lab/` em cada PC. Sem argumento imprime um JSON (o painel usa); `--tui` mostra o status no terminal. |
| `show_status.sh` | Abre a janela de status (`lab_status.py --tui`) na tela do PC (DISPLAY :0). |
| `provision.py` | Integra PCs novos: kit + artefatos copiados de um PC já pronto (HTTP na porta 8800) + `lab_setup.sh` + janela + `hosts.json`. Uso: `python provision.py 9.05=<ip> 9.07=<ip>` |

Ajuste de desempenho: `lab/tune_vllm.py` (uma configuração por execução, grava `~/dory-lab/tune/<label>.json`) e
`lab/tune_queue.sh <arquivo>` (fila; uma linha de argumentos por configuração; espera a GPU liberar entre execuções).

Requisitos no chefe: Python 3, OpenSSH do Windows, a chave `~/.ssh/id_ed25519_lab` (não versionada) e, para o Drive, rclone com o remote `gdrive:`.
