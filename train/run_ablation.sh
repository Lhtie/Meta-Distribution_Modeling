# echo "Margin = 0"  && 
# bash run_scorer.sh prob 0  && 
# echo "Margin = 0" > eval_prob.txt && 
# bash run_eval.sh 0 0 >> eval_prob.txt &&
# echo "Margin = 0.4"  && 
# bash run_scorer.sh prob 0.4  && 
# echo "Margin = 0.4" > eval_prob.txt && 
# bash run_eval.sh 0 0.4 >> eval_prob.txt &&
# echo "Margin = 0.6"  && 
# bash run_scorer.sh prob 0.6  && 
# echo "Margin = 0.6" >> eval_prob.txt && 
# bash run_eval.sh 0 0.6 >> eval_prob.txt &&
# echo "Margin = 0.8"  && 
# bash run_scorer.sh prob 0.8  && 
# echo "Margin = 0.8" >> eval_prob.txt && 
# bash run_eval.sh 0 0.8 >> eval_prob.txt &&
# echo "Margin = 1.0"  && 
# bash run_scorer.sh prob 1.0  && 
# echo "Margin = 1.0" >> eval_prob.txt && 
# bash run_eval.sh 0 1.0 >> eval_prob.txt &&
# echo "Margin = 1.6"  && 
# bash run_scorer.sh prob 1.6  && 
# echo "Margin = 1.6" >> eval_prob.txt && 
# bash run_eval.sh 0 1.6 >> eval_prob.txt &&
# echo "Margin = 2.4"  && 
# bash run_scorer.sh prob 2.4  && 
# echo "Margin = 2.4" >> eval_prob.txt && 
# bash run_eval.sh 0 2.4 >> eval_prob.txt &&
# echo "Margin = 4.2"  && 
# bash run_scorer.sh prob 4.2  && 
# echo "Margin = 4.2" >> eval_prob.txt && 
# bash run_eval.sh 0 4.2 >> eval_prob.txt &&
# echo "Margin = 6.4"  && 
# bash run_scorer.sh prob 6.4  && 
# echo "Margin = 6.4" >> eval_prob.txt && 
# bash run_eval.sh 0 6.4 >> eval_prob.txt &&
# echo "Margin = 8.4"  && 
# bash run_scorer.sh prob 8.4  && 
# echo "Margin = 8.4" >> eval_prob.txt && 
# bash run_eval.sh 0 8.4 >> eval_prob.txt &&
# echo "Margin = 10.2"  && 
# bash run_scorer.sh prob 10.2  && 
# echo "Margin = 10.2" >> eval_prob.txt && 
# bash run_eval.sh 0 10.2 >> eval_prob.txt
# echo "Margin = 12.8"  && 
# bash run_scorer.sh prob 12.8  &&
# echo "Margin = 12.8" >> eval_prob.txt && 
# bash run_eval.sh 0 12.8 >> eval_prob.txt

echo "Margin = 6.4, lr = 5e-6" &&
bash run_scorer.sh prob 6.4 5e-6 &&
echo "Margin = 6.4, lr = 5e-6" > eval_prob.txt &&
bash run_eval.sh 0 >> eval_prob.txt &&
bash run_eval.sh 1 >> eval_prob.txt &&
bash run_eval.sh 2 >> eval_prob.txt &&
echo "Margin = 6.4, lr = 1e-5" &&
bash run_scorer.sh prob 6.4 1e-5 &&
echo "Margin = 6.4, lr = 1e-5" >> eval_prob.txt &&
bash run_eval.sh 0 >> eval_prob.txt &&
bash run_eval.sh 1 >> eval_prob.txt &&
bash run_eval.sh 2 >> eval_prob.txt &&
echo "Margin = 6.4, lr = 3e-5" &&
bash run_scorer.sh prob 6.4 3e-5 &&
echo "Margin = 6.4, lr = 3e-5" >> eval_prob.txt &&
bash run_eval.sh 0 >> eval_prob.txt &&
bash run_eval.sh 1 >> eval_prob.txt &&
bash run_eval.sh 2 >> eval_prob.txt