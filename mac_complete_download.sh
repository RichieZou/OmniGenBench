#!/bin/bash
# 完整的Mac下载脚本（包含Git LFS安装）

echo "🚀 完整下载OmniGenome-52M模型"
echo "=============================="

# 1. 检查并安装Git LFS
echo "🔧 检查Git LFS..."
if ! command -v git-lfs &> /dev/null; then
    echo "📦 安装Git LFS..."
    
    # 尝试使用Homebrew安装
    if command -v brew &> /dev/null; then
        echo "🍺 使用Homebrew安装Git LFS..."
        brew install git-lfs
    else
        echo "❌ Homebrew未安装，请手动安装Git LFS:"
        echo "   1. 访问: https://git-lfs.github.io/"
        echo "   2. 下载Mac版本安装包"
        echo "   3. 或者安装Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "   4. 然后运行: brew install git-lfs"
        exit 1
    fi
else
    echo "✅ Git LFS已安装"
fi

# 2. 删除旧的不完整模型
echo "🗑️  删除旧模型..."
rm -rf ~/Downloads/OmniGenome-52M

# 3. 创建新目录并下载
echo "📁 创建下载目录..."
mkdir -p ~/Downloads/OmniGenome-52M
cd ~/Downloads/OmniGenome-52M

# 4. 克隆模型仓库
echo "📥 克隆模型仓库..."
git clone https://huggingface.co/yangheng/OmniGenome-52M .

# 5. 配置Git LFS
echo "🔧 配置Git LFS..."
git lfs install

# 6. 下载大文件
echo "📦 下载模型权重文件..."
git lfs pull

# 7. 检查下载结果
echo "📊 检查下载结果..."
ls -la

echo ""
echo "📏 关键文件大小:"
if [ -f "model.safetensors" ]; then
    size=$(du -h model.safetensors | cut -f1)
    echo "  📄 model.safetensors: $size"
    
    # 检查是否是真正的模型文件
    file_type=$(file model.safetensors)
    if [[ $file_type == *"data"* ]] || [[ $file_type == *"binary"* ]]; then
        echo "  ✅ 模型文件下载成功（二进制文件）"
    else
        echo "  ❌ 模型文件可能不完整（文本文件）"
    fi
else
    echo "  ❌ model.safetensors 不存在"
fi

echo ""
echo "🎉 下载完成！"
echo "📁 模型位置: ~/Downloads/OmniGenome-52M"
echo ""
echo "💡 下一步："
echo "   1. 检查model.safetensors文件大小（应该约200MB）"
echo "   2. 传输到服务器:"
echo "      scp -r ~/Downloads/OmniGenome-52M/* yz1033@your-server:/home/yz1033/OmniGenBench/models_cache/"
