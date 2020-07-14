## Composition-Based Multi-Relational Graph Convolutional Networks

**Note:** Source code là của tác giả [CompGCN](https://github.com/malllabiisc/CompGCN), tất cả các phần cải tiến, case-study đều sẽ dựa vào source code này.

### Yêu cầu
- Python 3.x
- Pytorch 1.5
- torch_scatter


torch_scatter sẽ được cài như sau:
```
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```
`${CUDA}` có thể thay thế bằng `cpu`, `cu92`, `cu101`, `cu102` tuỳ vào phiên bản cài đặt Pytorch trên cpu hay gpu.


Chi tiết xem thêm tại https://github.com/rusty1s/pytorch_scatter
- Các dependencies còn lại có thể cài đặt bằng dòng lệnh sau:
```
pip install -r requirements.txt
```
### Training phase:
- Chạy `sh preprocess.sh` để tiến hành giải nén dataset và tạo một số thư mục cần thiết.

- Chạy dòng lệnh dưới đây để tiến hành training:

  ```shell
  python run.py -name test_run -using_gat -gpu 0
  ```

Các arguments và ý nghĩa của chúng có thể được tìm thấy bằng lệnh `python run.py -h`

### Case-study
- Cách 1: Sử dụng model đã được train tại training phase để tiến hành dự đoán

**Lưu ý:** model cần được lưu với đuôi `.pth`, sử dụng arguments `-savefinal` ở training phase để lưu model dưới dạng file `.pth`. 
- Cách 2: Sử dụng pretrained model được cung cấp tại [đây](https://drive.google.com/file/d/1tJcEdrsHDQ7DtylLbhQJ3ZG7Z68-PSqn/view?usp=sharing). Download model vào thư muc `checkpoints`
- Chạy câu lệnh sau đây để khởi tạo ứng dụng Link Prediction
```
python case_study.py
```
