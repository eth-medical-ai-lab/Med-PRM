<p align="center">
  <img src="media/medprm-logo.png" alt="medprm" style="width: 250px;">
</p>

# Med-PRM: Medical Reasoning Models with Step-wise Guideline-verified Process Rewards


<p align="center">
    <a href="https://arxiv.org/pdf/"><img alt="Paper" src="https://img.shields.io/badge/Paper-arxiv-blue"></a>
    <a href="https://med-prm.github.io/"><img alt="Website" src="https://img.shields.io/badge/Website-online-brightgreen"></a>
    <a href="https://github.com/eth-medical-ai-lab/Med-PRM"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-blue"></a>
    <a href="https://med-prm.github.io/#citation-ref"><img alt="Citation" src="https://img.shields.io/badge/Citation-reference-yellow"></a>
    <a href="https://huggingface.co/dmis-lab/llama-3.1-medprm-reward-v1.0"><img alt="Reward Model" src="https://img.shields.io/badge/Reward%20Model-Hugging%20Face-orange"></a>
    <a href="https://huggingface.co/dmis-lab/llama-3.1-medprm-policy-v1.0"><img alt="Policy Model" src="https://img.shields.io/badge/Policy%20Model-Hugging%20Face-orange"></a>
</p>



### News 

* [May/30/2025] 📈 **Med-PRM** is the first 8B framework to achieve over 80% accuracy on MedQA.
* [May/30/2025] 🎉 🎊 🎉 We are thrilled to announce the first launch of **Med-PRM**, an innovative medical process reward model.


## 📖 Overview

- **MED-PRM** is a cutting-edge framework designed to enhance clinical decision-making by addressing errors in the reasoning process. It leverages retrieval-augmented generation (RAG) to verify each reasoning step against established medical knowledge bases, ensuring accuracy and reliability in medical diagnoses. MED-PRM is not intended to replace the expertise of healthcare professionals but to augment it, allowing them to focus on critical thinking and patient care while automating the verification of reasoning steps.
<p align="center">
  <img src="media/medprm-overview.png" alt="Overview of MED-PRM" style="width: 99%;">
</p>


## Models

| Policy Model       | Reward Model       | Policy Base Model | Reward Base Model | Policy Training Method | Reward Training Method | MedQA-4 | MedQA-5 |
|--------------------|--------------------|-------------------|-------------------|------------------------|------------------------|---------|---------|
| Med PRM Policy v1.0 | Med PRM Reward v1.0 | Llama 3.1 8B IT | Llama 3.1 8B IT | Rejection Sampling SFT | SFT | 80.5 | 78.2 |
| Llama 3.1 8B IT | Med PRM Reward v1.0 | Llama 3.1 8B IT | Llama 3.1 8B IT | - | SFT | TBD | TBD |

*Med-PRM-1.0 achieves state-of-the-art performance across multiple medical benchmarks.*


## 🖥️ Installation

### Python venv option

* We recommend using Python 3.12

1. **Clone the GitHub Repository**: Begin by cloning the repository using the command:

   ```bash
   git clone https://github.com/eth-medical-ai-lab/Med-PRM.git
   ```

2. **Navigate to the Project Directory**: Change into the project directory:

   ```bash
   cd Med-PRM
   ```

3. **Set Up a Virtual Environment**: Create and activate a virtual environment:

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

4. **Install Required Packages**: Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```


### Contact
Feel free to reach out to #1 email #2 email


**BibTeX Citation**: If you use Med-PRM in your research, please cite it using the following BibTeX entry:

   ```bibtex
   @misc{medprm2025,
     title={Med-PRM: Medical Reasoning Models with Step-wise Guideline-verified Process Rewards},
     author={Jaehoon Yun and Jiwoong Sohn and Jungwoo Park and Hyunjae Kim and Xiangru Tang and Daniel Shao and Yong Hoe Koo and Ko Minhyeok and Qingyu Chen and Mark Gerstein and Michael Moor and Jaewoo Kang},
     year={2025},
     month={May},
     url={}
   }
   ```

