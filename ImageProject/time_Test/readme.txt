在timeInformation.txt中可以找到opencv3中cpu和gpu处理形态学操作、边缘检测操作、模糊去噪操作的相关API函数。
运行环境：centos7.1+cuda7.5+opencv3.0 显卡nvidia GTX560
关于边缘检测和模糊去噪，其GPU运行速度至少比cpu运行速度高1倍（详细信息在timeInformation.txt中），而形态学操作，却是取决于cuda的版本，在相同显卡下，用cuda5.0的形态学操作明显GPU快于CPU， 但是在cuda7.5中，GPU居然比CPU还慢至少2倍，不知道nvidia优化了什么。。。
