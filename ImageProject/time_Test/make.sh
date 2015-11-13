#########################################
#         File name   :make.sh
#         Author      :liangkangkang
#         File desc   :
#         Mail        :liangkangkang@paag.com
#         Create time :2015-09-06
#########################################
#!/usr/bin/bash
#obj = `echo $1 | cut -d '.' -f1`
#g++ -Wall -std=c++0x $1 -o $obj
g++ -Wall -std=c++0x $1  `pkg-config --cflags --libs opencv` -o  `echo $1 | cut -d '.' -f1`
#./`echo $1 | cut -d '.' -f1`
