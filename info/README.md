In the current result table on [BAMBOO](https://github.com/flipz357/bamboo-amr-benchmark) (version 07/12/2023), we used these shell script commands (modify paths as needed, the commands should be exectuable by placing the `weisfeiler-leman-amr-metrics` directory in the `evaluation-suite` directory of BAMBOO)

For WLK:

```
CURR=$(pwd)
echo $CURR

for dat in sts sick para
do
    for task in reify main role_confusion syno
    do
    echo $dat $task
    python weisfeiler-leman-amr-metrics/src/main_wlk.py -a ../$dat/$task/tgt.test.amr -b ../$dat/$task/src.test.amr -round_decimals 16 > sim-predictions/$dat-wlkdefault-$task.txt
    done
done
```

For WWLK:

```
CURR=$(pwd)
echo $CURR

for dat in sts sick para
do
    for task in reify main role_confusion syno
    do
    echo $dat $task
    python weisfeiler-leman-amr-metrics/src/main_wlk_wasser.py -a ../$dat/$task/tgt.test.amr -b ../$dat/$task/src.test.amr -round_decimals 16 -stability_level 15 > sim-predictions/$dat-wwlkdefault-$task.txt
    done
done
```

For WWLK-train (WWLK theta):

```
CURR=$(pwd)
echo $CURR

for dat in sts sick para
do
    for task in reify main syno
    do
    echo $dat $task
    python weisfeiler-leman-amr-metrics/src/main_wlk_wasser_optimized.py -a_train ../$dat/$task/tgt.train.amr -b_train ../$dat/$task/src.train.amr -a_dev ../$dat/$task/tgt.dev.amr -b_dev ../$dat/$task/src.dev.amr -y_dev ../$dat/dev.y -a_test ../$dat/$task/tgt.test.amr -b_test ../$dat/$task/src.test.amr -y_train ../$dat/train.y > sim-predictions/$dat-wwlkthetadefault-$task.txt 
    done
#done

task=role_confusion
for dat in sts sick para
do
    python weisfeiler-leman-amr-metrics/src/main_wlk_wasser_optimized.py -a_train ../$dat/$task/tgt.train.amr -b_train ../$dat/$task/src.train.amr -a_dev ../$dat/$task/tgt.dev.amr -b_dev ../$dat/$task/src.dev.amr -a_test ../$dat/$task/tgt.test.amr -b_test ../$dat/$task/src.test.amr > sim-predictions/$dat-wwlkthetadefault-$task.txt
done
```
