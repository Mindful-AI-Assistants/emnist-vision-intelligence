
<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]**

<br><br>


#  <p align="center"> 2- 🧠 [Machine Learning]() / [EMNIST Vision Intelligence Project]()
### <p align="center"> Deep Learning Pipeline for Handwritten Character Recognition with PyTorch and Streamlit


<br><br>
<!-- ========= END REPO TITLE ========= -->


<br><br>

<!-- ========= START MAIN BADGE ========= -->
<p align="center" style="margin: 0;">
  <a href="https://identificados-de-letras-e-numeros.streamlit.app/" rel="noopener noreferrer">
    <img 
      src="https://img.shields.io/badge/Streamlit%20Interactive-EMNIST%20Vision%20Intelligence%20Dashboard-0f172a?style=for-the-badge&logo=streamlit&logoColor=white" 
      alt="Streamlit Interactive – EMNIST Vision Intelligence Dashboard"
      style="height: 38px; width: auto;"
    />
  </a>
</p>

<!-- ========= START SECONDARY BADGES ========= -->
<p align="center" style="margin: 0;">
  
  <a href="https://www.canva.com/design/DAHI8GmXZP8/Hxi6FpvjeKvf3em_iXlp2Q/edit" rel="noopener noreferrer">
    <img 
      src="https://img.shields.io/badge/Beautiful.ai-Deep%20Learning%20Presentation-0f766e?style=for-the-badge&logo=beautifulsoup&logoColor=white" 
      alt="Beautiful.ai Deep Learning Presentation"
      style="height: 32px; width: auto; margin-right: 8px;"
    />
  </a>

 <a href="#" target="_blank" rel="noopener noreferrer">
    <img 
      src="https://img.shields.io/badge/Model%20Analysis-Neural%20Network%20Report-134e4a?style=for-the-badge&logo=googleanalytics&logoColor=white&labelColor=022c22" 
      alt="Neural Network Analysis Report"
      style="height: 32px; width: auto;"
    />
  </a>

</p>
<br><br><br><br>
<!-- ========= END BADGE ========= -->

<!-- =========  START PUC HEADER GIF ========= -->

<p align="center">
   <img src="https://github.com/user-attachments/assets/791a69e2-d09a-429f-9257-f6667fff5c04 ">
 </p>


<br><br><br><br>
<!-- =========  END PUC HEADER GIF ========= -->


<!-- ========= START SPONSORT BADGE ========= -->
 <!--### <p align="center">  <img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="46">  -->

#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-%C2%B7%C2%B7%C2%B7%20Mindful%20AI%20Assistants%20%C2%B7%C2%B7%C2%B7-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)

<br><br><br>
<!-- ========= END SPONSORTBADGE ========= -->



<!-- ======================================= Start nstitucional INFO ===========================================  -->
 <!--### <p align="center">  <img src="https://github.githubassets.com/images/icons/emoji/octocat.png" width="46">  -->
[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP  Humanistic AI & Data Science • 5º Semestre • 2026 <br>
[**School:**]() Faculty of Interdisciplinary Studies  <br>
[Course Repo:]() INTEGRATED PROJECT: MACHINE LEARNING  <br>
**Professor:**  [✨ Rooney Ribeiro Albuquerque Coelho](https://www.linkedin.com/in/rooney-coelho-320857182/)  <br>
**Authors**:**  [Fabiana ⚡️ Campanari](https://linktr.ee/fabianacampanari) e [Perdro Vyctor Almeida]()  <br>  <br>

<br><br>

#

<br><br>
<!-- ========= END Institucional INFO ========= -->

<!-- ========= START BADGES PYRAMID ========= -->

<p align="center">
  <img src="https://img.shields.io/badge/Python-AI%20Pipeline-0f172a?style=for-the-badge&logo=python&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-101f2f?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CNN-Computer%20Vision-112a3a?style=for-the-badge&logo=tensorflow&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-124050?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorBoard-Training%20Metrics-134a4a?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/EMNIST-47%20Classes-134e4a?style=for-the-badge&logo=tensorflow&logoColor=white" />
</p>

<!-- ========= END BADGES PYRAMID ========= -->



<br><br>










## MLOps Pipeline Architecture


<br><br>


```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#020617",
    "primaryColor": "#020617",
    "primaryBorderColor": "#2dd4bf",
    "primaryTextColor": "#ffffff",
    "lineColor": "#2dd4bf",
    "secondaryColor": "#020617",
    "tertiaryColor": "#020617",
    "clusterBkg": "#020617",
    "clusterBorder": "#2dd4bf",
    "fontFamily": "Inter, monospace",
    "fontSize": "16px"
  }
}}%%

flowchart TB

%% =========================
%% STYLE
%% =========================
classDef clean fill:#020617,stroke:#2dd4bf,color:#ffffff,stroke-width:1px;

%% =========================
%% PIPELINE
%% =========================

A["EMNIST Balanced Dataset"]:::clean

B["Handwritten Character Samples"]:::clean

C["Image Preprocessing"]:::clean

D["Grayscale Conversion"]:::clean

E["Resize to 28×28"]:::clean

F["Tensor Transformation"]:::clean

G["Python AI Pipeline"]:::clean

H["PyTorch Deep Learning Model"]:::clean

I["Perceptron Layers"]:::clean

J["Convolutional Neural Networks"]:::clean

K["Dropout Regularization"]:::clean

L["Backpropagation & Optimization"]:::clean

M["TensorBoard Monitoring"]:::clean

N["Accuracy & Loss Evaluation"]:::clean

O["OpenCV Real-Time Processing"]:::clean

P["Character Prediction Engine"]:::clean

Q["Softmax Probability Output"]:::clean

R["Streamlit Interactive Dashboard"]:::clean

S["Drawing Canvas Interface"]:::clean

T["Final AI Inference"]:::clean

%% =========================
%% FLOW
%% =========================

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G

G --> H
H --> I
I --> J
J --> K
K --> L

L --> M
M --> N

N --> O
O --> P
P --> Q

Q --> R
R --> S
S --> T

linkStyle default stroke:#2dd4bf,stroke-width:1px;
```
















































<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>


<!-- ======================================= Start DEFAULT Footer ===========================================  -->

<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br>


#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />


<br><br>

<p align="center">  ────────────── ⊹🔭๋ ──────────────

<!--
<p align="center">  ────────────── 🛸๋*ੈ✩* 🔭*ੈ₊ ──────────────
-->

<br>

<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>
  

  
#
 
##### <p align="center">Copyright 2026 Mindful-AI-Assistants. Code released under the  [MIT license.](https://github.com/Mindful-AI-Assistants/CDIA-Entrepreneurship-Soft-Skills-PUC-SP/blob/21961c2693169d461c6e05900e3d25e28a292297/LICENSE)




<!-- ======================================= End  DEFAULT Footer ===========================================  -->














