# python3 experiment.py --dim=1
# python3 experiment.py --dim=2
# python3 experiment.py --dim=3

for d in 1 2 3 4 5
do
    taskset -c 0-50 python3 experiment.py --dim=$d
done