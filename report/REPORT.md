# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** Python và PyTorch dùng theo môi trường Colab của lần chạy; Ultralytics 8.4.145.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không thay đổi code, checkpoint hoặc cấu trúc output của notebook nguồn. Detection và instance segmentation dùng ngưỡng confidence mặc định 0.35 theo notebook.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (class_id, class_name, rank, score, taxonomy_name): class_id=468, class_name="cab", rank=1, score=0.510915, taxonomy_name="ImageNet-1K".
- Record này mô tả toàn ảnh như thế nào? Đây là prediction cấp ảnh: mô hình xếp lớp cab cao nhất cho toàn bộ ảnh traffic, không phải nhãn cho một object riêng lẻ trong ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Danh sách lớp được xác định bởi taxonomy/dataset mà checkpoint classification được huấn luyện trên. Trong bài này checkpoint yolo11n-cls.pt dùng taxonomy ImageNet-1K.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? class_id thuận tiện cho máy xử lý và đối chiếu nhất quán; class_name giúp con người hiểu ý nghĩa lớp; taxonomy_name cho biết ID và tên lớp thuộc hệ nhãn nào, tránh nhầm cùng một ID giữa các taxonomy khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline phải quy định rõ tiêu chí chọn nhãn cấp ảnh, ví dụ chọn chủ thể chính/nội dung chính hay cho phép multi-label. Trường hợp không xác định được chủ thể chính thì cần chuyển reviewer hoặc escalation thay vì tự suy đoán.
- Vì sao model score không phải ground truth? score=0.510915 chỉ thể hiện mức độ tự tin của mô hình đối với prediction cab. Ground truth là nhãn tham chiếu được tạo/xác nhận theo guideline và quy trình annotation/QC của con người; prediction có score cao vẫn có thể sai.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): class_name="person", score=0.912625, bbox_xyxy=[385.33, 69.24, 498.92, 348.92], bbox_width=113.58, bbox_height=279.68 pixel, sample_id="kitchen".
- Diễn giải vị trí box bằng lời: bbox_xyxy=[x_min, y_min, x_max, y_max] dùng đơn vị pixel và gốc tọa độ ở góc trên trái. Box của person bắt đầu khoảng tại (385.33, 69.24) và kết thúc tại (498.92, 348.92), bao quanh người đứng ở vùng giữa-phải của ảnh.
- So sánh số prediction ở hai threshold: với sample kitchen, tại threshold 0.35 có 11 prediction. Nếu nâng threshold lên 0.60, còn 6 prediction có score từ 0.60 trở lên. Như vậy threshold cao hơn đã loại 5 prediction confidence thấp hơn.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp giữ nhiều prediction hơn nên tăng khả năng bao phủ object nhưng cũng làm reviewer phải xem nhiều prediction yếu/sai hơn. Threshold cao giảm khối lượng review nhưng có nguy cơ bỏ mất object thật có confidence thấp. Threshold chỉ là cấu hình lọc prediction, không phải quy tắc tạo ground truth.
- Đề xuất một quy tắc box chặt: Bounding box phải bao hết phần nhìn thấy rõ của object mục tiêu, ôm sát biên ngoài của vật thể, hạn chế nền thừa và không cố tình bao thêm object khác.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Hình kitchen có một người ở mép trái bị cắt khỏi khung hình và nhiều vật dụng/bát đặt gần nhau. Guideline cần quy định box chỉ bao phần nhìn thấy hay có được ước lượng phần bị che/cắt; nếu ranh giới không đủ rõ hoặc mức che khuất vượt quy định thì annotator cần chuyển reviewer/escalation.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): instance_id="kitchen-001", class_name="person", score=0.899318, polygon_point_count=348; một phần đầu của polygon_xy là [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]].
- Polygon bổ sung chi tiết gì so với box? Box chỉ cho một hình chữ nhật bao quanh object; polygon chứa nhiều điểm theo biên nên mô tả hình dạng và vùng pixel của instance chi tiết hơn, đặc biệt với vật thể có hình dạng không chữ nhật.
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id dùng để phân biệt từng instance trong output, kể cả khi nhiều object có cùng class. Ví dụ kitchen-001 là một instance riêng. Đây không phải class_id, không phải ID của người/vật ngoài đời và cũng không phải tracking ID qua nhiều frame/ảnh.
- Đề xuất một quy tắc biên mask: Mask/polygon phải bám theo phần pixel nhìn thấy thực sự thuộc object, hạn chế ăn sang nền hoặc object bên cạnh và không tự suy diễn phần bị che khuất nếu guideline không yêu cầu
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Trong ảnh kitchen, nhiều vật thể trên bàn tiếp xúc/gần nhau và người ở mép trái bị cắt. Guideline cần nói rõ cách tách biên giữa các instance chạm nhau, xử lý vùng biên mờ và có annotate phần bị che/cắt hay không. Nếu không thể xác định biên ổn định từ ảnh thì cần escalation cho reviewer.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`
Ảnh thô được đưa vào quy trình gán nhãn theo guideline. Annotator tạo ground truth theo đúng định dạng của từng tác vụ. Ground truth được dùng cho huấn luyện/đánh giá mô hình. Mô hình sau đó tạo prediction; prediction phải được QC bằng cách đối chiếu với ảnh, guideline và ground truth. Prediction sai hoặc trường hợp mơ hồ cần review/rework, không được tự coi prediction là nhãn đúng.

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cấp ảnh theo taxonomy; dự án khác có thể quy định multi-label | Ảnh có thể chứa nhiều chủ thể nên nhãn cấp ảnh dễ mơ hồ; sample traffic được model xếp cab với score chỉ khoảng 0.51 | Chọn nhãn theo tiêu chí chủ thể/nội dung chính đã ghi trong guideline; đánh dấu/escalate nếu không rõ | Kiểm tra nhãn có đúng taxonomy, đúng phạm vi toàn ảnh và có tuân thủ guideline không |
| Phát hiện vật thể | Một class + một bbox_xyxy cho mỗi object/instance | Object bị cắt mép, che khuất; nhiều bowl gần nhau; threshold thay đổi số prediction | Gán class và vẽ box ôm sát từng object nhìn thấy theo guideline; không dùng confidence làm ground truth | Kiểm tra object bị bỏ sót/thừa, class sai, box quá rộng/hẹp, cách xử lý occlusion/cut-off và tính nhất quán giữa annotator |
| Instance segmentation | Một class + mask/polygon cho mỗi instance | Biên giữa các vật thể chạm/gần nhau khó xác định; vật thể bị che/cắt làm polygon mơ hồ | Vẽ mask/polygon bám biên từng instance theo guideline và tách các object cùng lớp thành instance riêng | Kiểm tra class, instance_id, biên mask, vùng ăn sang nền/object khác, lỗ hổng/bỏ sót và các ca cần escalation |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng các ảnh mẫu công khai đúng phạm vi bài thực hành và không đưa họ tên, MSSV, email, số điện thoại, dữ liệu khách hàng hoặc dữ liệu nhạy cảm khác vào REPORT.md hay output công khai.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: mentor/giảng viên phụ trách bài thực hành trước khi tiếp tục xử lý hoặc đưa dữ liệu lên repository.



## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
