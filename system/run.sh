nohup python3 main.py -algo FDFed -lr 0.01 -al 0.001 -m effnet -mn effnet -lam 0.35 -th 2 -nc 3 -data nihchestxray -t 5 -go experiment -gpu 0 > FDFed_nihchestxray_3_effnet.log 2>&1 &
wait
nohup python3 main.py -algo FedBABU -lr 0.01 -al 0.001 -m effnet -mn effnet -nc 3 -data nihchestxray -t 5 -fts 5 -go experiment -gpu 0 > FedBABU_nihchestxray_3_effnet.log 2>&1 &
wait
nohup python3 main.py -algo FedPer -lr 0.01 -m effnet -mn effnet -nc 3 -data nihchestxray -t 5 -go experiment -gpu 0 > FedPer_nihchestxray_3_effnet.log 2>&1 &
wait
nohup python3 main.py -algo FedRep -lr 0.01 -m effnet -mn effnet -nc 3 -data nihchestxray -t 5 -go experiment -gpu 0 > FedRep_nihchestxray_3_effnet.log 2>&1 &
wait
nohup python3 main.py -algo FedPav -lr 0.01 -m effnet -mn effnet -nc 3 -data nihchestxray -t 5 -go experiment -gpu 0 > FedPav_nihchestxray_3_effnet.log 2>&1 &
wait
nohup python3 main.py -algo Local -lr 0.01 -m effnet -mn effnet -nc 3 -data nihchestxray -t 5 -go experiment -gpu 0 > Local_nihchestxray_3_effnet.log 2>&1 &


wait
nohup python3 main.py -algo FDFed -lr 0.01 -al 0.005 -m mobilenet -mn mobilenet -lam 0.28 -th 3 -nc 5 -data cifar10 -t 5 -go experiment -gpu 0 > FDFed_cifar10_5_mobilenet.log 2>&1 &
wait
nohup python3 main.py -algo FedBABU -lr 0.01 -al 0.001 -m mobilenet -mn mobilenet -nc 5 -data cifar10 -t 5 -fts 5 -go experiment -gpu 0 > FedBABU_cifar10_5_mobilenet.log 2>&1 &
wait
nohup python3 main.py -algo FedPer -lr 0.01 -m mobilenet -mn mobilenet -nc 5 -data cifar10 -t 5 -go experiment -gpu 0 > FedPer_cifar10_5_mobilenet.log 2>&1 &
wait
nohup python3 main.py -algo FedRep -lr 0.01 -m mobilenet -mn mobilenet -nc 5 -data cifar10 -t 5 -go experiment -gpu 0 > FedRep_cifar10_5_mobilenet.log 2>&1 &
wait
nohup python3 main.py -algo FedPav -lr 0.01 -m mobilenet -mn mobilenet -nc 5 -data cifar10 -t 5 -go experiment -gpu 0 > FedPav_cifar10_5_mobilenet.log 2>&1 &
wait
nohup python3 main.py -algo Local -lr 0.01 -m mobilenet -mn mobilenet -nc 5 -data cifar10 -t 5 -go experiment -gpu 0 > Local_cifar10_5_mobilenet.log 2>&1 &
