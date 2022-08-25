0413
advanced 0409 experiment.
I train the model with same seed with get the same result model at the single "python ...py"
However, if I use another command time "python ...py". I got 3 time same model but different from the previous trainingn command "python ...p".
Conclusion:
Setting “torch.backends.cudnn.benchmark = True” “torch.backends.cudnn.deterministic = False” can ensure to get the same training result at the cost of training speed about 30~50%
The cuda will pick the fastest algo of convolution in the process.
If I use the accelerating feature, the model output will be slightly different at fowrward propagation, which will lead to the different loss.
However, if I make cuda to choose the same algo of convolution. I can get the same result model even in the each iteraion. That's a trade-off.
Pytorch officailly say they cannot promise to reporduce the training process and result model.

0409
resolve 0405 experiment questions.
training 5 rounds with the same initail weight.
conclusion:
I got the same same nas mode and same trained model. 
Without a doubt, testing accuracy is all the same.



0405
two parallel innercell, five layers
conclusion:
It's strange that the same initail weight will get different model and result accuracy. Please make sure there is nothing wrong during training proceudre.




3/10
pick the operation in a cell according to two top-2 alphas
## procedure
1. commnad "python train_nas_5cell.py"
train_nas_5cell.py is a more readable file modified from train_search_5cell.py
2.  command "python decode_pdarts.py"
用pickSecondMax()來選出 ./alpha_pdart_nodrop內的第二大alphas
3. command "python retrain_5cell.py"
4. command "python test.py"


