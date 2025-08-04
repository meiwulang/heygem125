# 模型下载
`bash download.sh`

安装 **Python 3.8**。然后，使用 pip 安装项目依赖项  
```bash
conda create -n heygem python=3.8 -y # 创建环境
conda activate heygem # 进入环境
python -V #查看是否为3.8版本
```
安装cuda相关配置
```bash
conda install cudatoolkit=11.8 cudnn=8.9.2 #安装cuda 11.8版本
pip install onnxruntime-gpu==1.16.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
然后输入以下命令安装
```bash

pip install -r requirements.txt 
# 指定阿里云镜像
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

```
安装ffmpeg等软件依赖，输入以下命令安装 ffmpeg版本一定要高于4.0
```
sudo apt update
sudo apt install ffmpeg -y
```
# 启动接口版本
`python app.py`

# 启动web版本
`python gradio_web.py`
---



# 反编译说明
转换动作已经省略，可以直接反编译pyc文件

## 安装python3.8.19
`conda create -n dpy python=3.8.19`

安装反编译工具 

`pip install decompyle3`


脚本如下
```bash
  find . -name "*.cpython-38.pyc" | while read pyc_file; do
    dir=$(dirname "$pyc_file")
    base_name=$(basename "$pyc_file" .cpython-38.pyc)
    target_py="${dir}/../${base_name}.py"
    decompyle3 "$pyc_file" > "$target_py"
    echo "反编译完成: $pyc_file -> $target_py"
  done
```
- decompyle3遇到复杂异常情况也无法反编译 需要用到pycdc
- pycdc 
```
git clone https://github.com/zrax/pycdc
cd pycdc
cmake .
make
./pycdc your_file.pyc > output.py
```
我的完整测试脚本
```
cd /Users/wangbin/Documents/heygem_code/code/pycdc 
../service/__pycache__/trans_dh_service.cpython-38.pyc >../service/trans_dh_service_pycdc.py
```
# TODO
还有一部分无法还原，需要从汇编语言转到 python 