# DeepConText112
Deep continual learning pipelines for out-of-hospital medical emergencies text classification under the presence of dataset shifts.

## Description
This repository contains code developed for implementing and evaluating various deep continual learning text classification pipelines. It also facilitates the examination of temporal dataset shifts. The associated code accompanies a manuscript published at *Computers in Biology and Medicine*.

The study considers three types of temporal dataset shifts:
- Temporal prior probability shifts.
- Temporal covariate shift.
- Temporal concept shift.

For deep continual text classification baselines, the following approaches were evaluated:
- Static modeling.
- Single fine-tuning.
- Joint training.

The deep continual learning text pipelines implemented include:
- Cumulative learning.
- Continual fine-tuning.
- Replay.
- Synaptic intelligence.

This code has been tested on a protected database of out-of-hospital medical emergencies from the Valencian Community (Spain), encompassing a total of 1 982 746 independent medical incidents. 

Furthermore, this code can serve as a template for numerous other applications that require a deep continual text classification approach.

## Citation
The methods and evaluation results are published in the following article, please cite it if you use this code:

Pablo Ferri, Vincenzo Lomonaco, Lucia C. Passaro, Antonio Félix-De Castro, Purificación Sánchez-Cuesta, Carlos Sáez and Juan M García-Gómez. "Deep continual learning for medical call incidents text classification under the presence of dataset shifts". Computers in Biology and Medicine, 108548 (2024). https://doi.org/10.1016/j.compbiomed.2024.108548.

## Credits
- **Main developer**: Pablo Ferri, Ph.D.
- **Authors**: Pablo Ferri (UPV), Vincenzo Lomonaco (UNIPI), Lucia C.Passaro (UNIPI), Antonio Félix-De Castro (GVA), Purificación Sánchez-Cuesta (GVA), Carlos Sáez (UPV) and Juan M García-Gómez (UPV)

Copyright: 2024 - Biomedical Data Science Lab, Universitat Politècnica de València, Spain (UPV)

## Acknowledgements
This project was supported by the Ministry of Science, Innovation, and Universities of Spain through the FPU18/06441 program and partially funded by the PNRR-M4C2-Investment 1.3, Extended Partnership PE00000013-FAIR (Future Artificial Intelligence Research)-Spoke 1 Human-centered AI, financed by the European Commission under the NextGeneration EU program.

