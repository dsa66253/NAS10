@REM timeout 600
python refresh.py

python trainNas.py
python mydecode_pdarts.py
python retrainNas.py


python test.py
python plotAcc.py
python drawAlpha.py

@REM python trainAlex.py
@REM python testAlex.py
@REM python plotAcc.py
