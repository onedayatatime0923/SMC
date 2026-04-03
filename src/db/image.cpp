#include "image.hpp"

NNV_NAMESPACING_START

int CImage::Run(int argc, char **argv) {
    ArgumentParser program("parse");

    program.AddArgument("image")
        .Help("taken the image from the path as the model input.")
        .Nargs(1);

    bool res;
    try {
    res = program.ParseArgs(argc, argv);
    }
    catch (const runtime_error& err) {
        std::cout << err.what() << std::endl;
        return 1;
    }
    if (res) return 0;

    Load(program.Get<string>("image"));

    return 0;
};

void CImage::Load(const string & image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    data_ = torch::from_blob(image.data, { image.rows * image.cols }, torch::kByte).to(torch::kFloat64) / 255;
}

NNV_NAMESPACING_END
