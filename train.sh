#!/bin/sh
# REM timeout 600


# python3 handleNextExperiment.py
# python3 trainNas.py
# python3 mydecode_pdarts.py
python3 retrainNas.py
python3 test.py
python3 plotAcc.py
# python3 drawAlpha.py
python3 handleCurExperiment.py
python3 refresh.py

# python3 handleNextExperiment.py
# python3 refresh.py
# python3 trainNas.py
# python3 mydecode_pdarts.py
# python3 retrainNas.py
# python3 test.py
# python3 plotAcc.py
# python3 drawAlpha.py
# python3 handleCurExperiment.py

# python3 handleNextExperiment.py
# python3 refresh.py
# python3 trainNas.py
# python3 mydecode_pdarts.py
# python3 retrainNas.py
# python3 test.py
# python3 plotAcc.py
# python3 drawAlpha.py
# python3 handleCurExperiment.py



