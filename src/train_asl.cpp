#include "../include/core/matrix.h"
#include "../include/core/utils.h"
#include "../include/layers/lstm.h"
#include "../include/layers/dense.h"
#include "../include/layers/softmax.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

// --- Config ---
const int JOINTS = 21;
const int COORDS = 3; // x, y, z
const int EPOCHS = 50;
const int HIDDEN_SIZE = 64;

struct ASLSample {
    std::vector<Matrix> sequence; // 21 steps of (3x1)
    int label;
};

// --- Normalization Logic ---
// CRITICAL: Makes the model invariant to where your hand is on the screen
void normalize_hand(std::vector<Matrix>& seq) {
    if (seq.empty()) return;
    
    // 1. Shift to Origin: Wrist (Index 0) becomes (0,0,0)
    Matrix wrist = seq[0]; 
    double max_dist = 0.0;

    for (auto& joint : seq) {
        joint = joint - wrist; // Relative position
        
        // Calculate distance from wrist to find Scale Factor
        double dist = std::sqrt(joint(0,0)*joint(0,0) + joint(1,0)*joint(1,0) + joint(2,0)*joint(2,0));
        if (dist > max_dist) max_dist = dist;
    }

    // 2. Scale to Unit Size (0.0 to 1.0 range)
    if (max_dist > 0) {
        for (auto& joint : seq) {
            joint = joint * (1.0 / max_dist);
        }
    }
}

// --- Robust CSV Loader ---
std::vector<ASLSample> load_data(const std::string& filename, int& num_classes) {
    std::vector<ASLSample> dataset;
    std::ifstream file(filename);
    std::string line;
    
    // Auto-Map: Converts "A" -> 0, "B" -> 1, "Space" -> 2
    std::map<std::string, int> label_map;
    int next_label_id = 0;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return dataset;
    }

    // 1. Skip Header
    std::getline(file, line);

    std::cout << "Loading " << filename << "..." << std::endl;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> raw_row;

        // Split line by comma
        while (std::getline(ss, segment, ',')) {
            raw_row.push_back(segment);
        }

        // Validation: Expect 63 coords + 1 label = 64 columns
        if (raw_row.size() < 64) continue;

        ASLSample s;
        
        // 2. Parse Coordinates (First 63 columns)
        try {
            for (int i = 0; i < JOINTS; i++) {
                Matrix joint(3, 1);
                // raw_row index: i*3 + 0 (x), +1 (y), +2 (z)
                joint(0, 0) = std::stod(raw_row[i * 3 + 0]);
                joint(1, 0) = std::stod(raw_row[i * 3 + 1]);
                joint(2, 0) = std::stod(raw_row[i * 3 + 2]);
                s.sequence.push_back(joint);
            }
        } catch (...) {
            continue; // Skip rows with bad numbers
        }

        // 3. Parse Label (The LAST column)
        std::string label_str = raw_row.back();
        
        // Clean whitespace
        label_str.erase(std::remove(label_str.begin(), label_str.end(), '\r'), label_str.end());
        label_str.erase(std::remove(label_str.begin(), label_str.end(), '\n'), label_str.end());

        // Assign ID
        if (label_map.find(label_str) == label_map.end()) {
            label_map[label_str] = next_label_id++;
            std::cout << "Found New Class: " << label_str << " -> ID " << label_map[label_str] << "\n";
        }
        s.label = label_map[label_str];

        // 4. Normalize
        normalize_hand(s.sequence);
        
        dataset.push_back(s);
    }
    
    num_classes = next_label_id;
    std::cout << "Loaded " << dataset.size() << " samples. Total Classes: " << num_classes << std::endl;
    return dataset;
}

int main() {
    srand(42);
    int num_classes = 0;
    
    // 1. Load Data
    std::vector<ASLSample> data = load_data("asl_landmarks_final.csv", num_classes);
    
    if (data.empty()) {
        std::cerr << "CRITICAL ERROR: Data loaded is empty. Check if file exists." << std::endl;
        return 1;
    }

    // 2. Architecture
    // Input: 3 (x,y,z) | Time Steps: 21 (joints) | Hidden: 64
    LSTM lstm(3, HIDDEN_SIZE);
    
    // Dense expects 1x64 Input (Row Vector), Output: num_classes
    Dense dense(HIDDEN_SIZE, num_classes);
    Softmax softmax;
    
    double lr = 0.001;

    std::cout << "Starting Training on ASL Data..." << std::endl;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0;
        int correct = 0;
        
        // Simple Shuffle
        std::random_shuffle(data.begin(), data.end());

        for (const auto& sample : data) {
            // Target One-Hot
            Matrix y_true = Matrix::zeros(1, num_classes);
            y_true(0, sample.label) = 1.0;

            // --- Forward ---
            // 1. LSTM (Sequence of 21) -> Returns Vector of (Hidden x 1)
            std::vector<Matrix> lstm_out = lstm.forward_pass(sample.sequence);
            Matrix h_last = lstm_out.back(); // Shape (64, 1)

            // 2. Transpose for Dense (64, 1) -> (1, 64)
            Matrix dense_in = h_last.transpose();

            // 3. Dense -> Softmax
            Matrix logits = dense.forward_pass(dense_in);
            Matrix probs = softmax.forward_pass(logits);

            // --- Backward ---
            // 1. Loss Gradient (P - Y)
            Matrix d_logits = probs - y_true;

            // 2. Dense Backward (Updates itself)
            // Returns (1, 64)
            Matrix d_h_T = dense.backward_pass(d_logits, lr);
            
            // 3. Transpose back for LSTM (64, 1)
            Matrix d_h = d_h_T.transpose();

            // 4. LSTM Backward
            std::vector<Matrix> d_seq(JOINTS, Matrix::zeros(HIDDEN_SIZE, 1));
            d_seq.back() = d_h; // Gradient only at last step

            lstm.backward_pass(d_seq);
            
            // 5. Update LSTM
            lstm.update(lr);

            // --- Stats ---
            total_loss += -std::log(probs(0, sample.label) + 1e-9);
            
            // Argmax
            int pred = 0;
            double max_p = -1.0;
            for(int k=0; k<num_classes; k++) {
                if(probs(0,k) > max_p) { max_p = probs(0,k); pred = k; }
            }
            if(pred == sample.label) correct++;
        }

        std::cout << "Epoch " << epoch 
                  << " | Loss: " << total_loss / data.size() 
                  << " | Acc: " << (double)correct / data.size() * 100.0 << "%" << std::endl;
    }

    return 0;
}