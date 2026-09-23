**active virtual environment**
sesame_rl\Scripts\active (windows)
source sesame_rl/bin/activate (ubuntu)
**python package list**

*Mujoco*

物理引擎的交互式渲染器

*Gymnasium*

Gymnasium 是 OpenAI 開發的 Gym 庫的維護分支，它提供了豐富的強化學習環境，讓研究人員和開發者能夠輕鬆地測試和比較各種強化學習算法。
<https://pypi.org/project/gymnasium/>

*stable-baselines3*

Stable Baselines3 是一套在 PyTorch 中可靠實現的強化學習算法。
<https://stable-baselines3.readthedocs.io/en/master/guide/install.html>

**Test**
actuator test
verisoncheck
view_sesame

**Git Hub**
*UPLOAD*
[ 工作區 (Working Directory) ]
       │
       │  1. git add .  (把修改或新檔案放到「暫存區」)
       ▼
[ 暫存區 (Staging Area) ]
       │
       │  2. git commit -m "訊息"  (把暫存區的內容「打包成版本」)
       ▼
[ 本地儲存庫 (Local Repository) ]
       │
       │  3. git push  (上傳到 GitHub)
       ▼
[ GitHub (Remote) ]

*DOWNLOAD*
git pull

## Project Stages

### 001 — Actuator Mapping / Model Setup

建立 Sesame 機器人的 MuJoCo 基礎模型，確認 8 個致動器的 joint mapping、順序、方向與初始站立姿態，作為後續模擬與控制的基礎。

### 002 — Firmware Gait Baseline

將原始 firmware gait 邏輯重現在 MuJoCo 中，包含步態時序、servo dynamics、接觸與運動診斷，用來建立未經學習的 locomotion baseline，並分析 lateral drift、yaw drift 與左右接觸差異。

### 003 — Reinforcement Learning

將 MuJoCo 模型封裝成 Gymnasium RL environment，使用 PPO & SAC 學習 8 維連續關節控制。此階段比較 Random、Firmware 與 learned policy，並透過 reward ablation 改善 forward locomotion、lateral motion 與 heading/yaw stability。

V3 採用 SAC，在與 V2 PPO 相同的 MuJoCo environment、observation/action space 與 Reward V2 設定下進行比較。評估結果顯示，V3 在保留或提升 forward locomotion 的同時，降低 lateral motion 與 yaw rate，並改善 directional ratio 與 path efficiency。這表示 SAC policy 在目前 Sesame locomotion task 中，不只提高推進能力，也改善了運動方向性與 heading/yaw stability。

因此，V2 → V3 的改善不能只由 episode reward 判斷，而應由 forward velocity、lateral velocity、yaw rate、directional ratio、path efficiency 與 stability metrics 共同驗證。
