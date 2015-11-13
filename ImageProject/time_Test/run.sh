#########################################################################
# File Name: run.sh
# Author: che fang
# mail: chefang.com
# Created Time: 2015年10月28日 星期三 14时02分27秒
#########################################################################
#!/bin/bash

c=`echo $1 | cut -d . -f1`
g++ $1 `pkg-config --cflags --libs opencv` -o $c

