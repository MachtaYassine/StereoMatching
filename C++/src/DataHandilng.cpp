#include <iostream>
#include <fstream>
#include <vector>



// Define a global constant for the image path
const std::string IMAGE_PATH = "/path/to/your/images/";

class ImageLoader {
public:
    ImageLoader(const std::string& name) : name_(name) {}

    bool LoadImages(std::vector<std::vector<uint8_t>>& left_images, std::vector<std::vector<uint8_t>>& right_images) {
        std::string left_image_path = IMAGE_PATH + name_ + "_left.png";
        std::string right_image_path = IMAGE_PATH + name_ + "_right.png";

        if (!ReadImage(left_image_path, left_images) || !ReadImage(right_image_path, right_images)) {
            return false;
        }

        return true;
    }

    bool SaveImages(const std::string& output_dir, const std::vector<std::vector<uint8_t>>& left_images, const std::vector<std::vector<uint8_t>>& right_images) {
        if (!WriteImage(output_dir + "/" + name_ + "_left.png", left_images) || !WriteImage(output_dir + "/" + name_ + "_right.png", right_images)) {
            return false;
        }

        return true;
    }

private:
    std::string name_;

    bool ReadImage(const std::string& file_path, std::vector<std::vector<uint8_t>>& image_data) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open image file: " << file_path << std::endl;
            return false;
        }

        // Read the file into a vector
        std::vector<uint8_t> data(
            (std::istreambuf_iterator<char>(file)),
            (std::istreambuf_iterator<char>())
        );

        if (data.empty()) {
            std::cerr << "Empty image file: " << file_path << std::endl;
            return false;
        }

        // Store the image data
        image_data.push_back(data);

        return true;
    }

    bool WriteImage(const std::string& file_path, const std::vector<std::vector<uint8_t>>& image_data) {
        std::ofstream file(file_path, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to create image file: " << file_path << std::endl;
            return false;
        }

        for (const auto& data : image_data) {
            file.write(reinterpret_cast<const char*>(data.data()), data.size());
        }

        return true;
    }
};

int main() {
    std::string image_name = "Road";  // Image name

    ImageLoader image_loader(image_name);
    std::vector<std::vector<uint8_t>> left_images, right_images;

    if (image_loader.LoadImages(left_images, right_images)) {
        // Images are loaded as binary data (vector of vectors of bytes)
        // You can process or display them as needed

        // Save the loaded images
        std::string output_dir = "output_directory";
        if (image_loader.SaveImages(output_dir, left_images, right_images)) {
            std::cout << "Images saved successfully." << std::endl;
        } else {
            std::cerr << "Failed to save images." << std::endl;
        }
    }

    return 0;
}
