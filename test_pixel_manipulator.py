import os
import cv2
import numpy as np
from pixel_watermark import PixelWatermark
from pixel_manipulator import PixelManipulator
from sipi_database_manager import sipi_manager
from app import app

def run_tests():
    print("=== STARTING ADVANCED PIXEL MANIPULATION & GRAYSCALE PERCENTAGE TESTS ===")
    
    # 1. Create a test RGB image
    test_img_path = "uploads/test_sample_image.png"
    os.makedirs("uploads", exist_ok=True)
    
    # Create 80x80 synthetic color image
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    for y in range(80):
        for x in range(80):
            img[y, x] = [int((x/80)*255), int((y/80)*255), int(((x+y)/160)*255)]
    
    cv2.imwrite(test_img_path, img)
    print(f"Created test image: {test_img_path} (80x80 = 6400 pixels)")

    # 2. Embed invisible watermark
    wm = PixelWatermark(strength=5)
    embed_ok = wm.embed_in_lsb(test_img_path, "Capstone!")
    print(f"LSB Watermark Embedded: {embed_ok}")
    extracted = wm.extract_lsb(test_img_path)
    print(f"Extracted initial watermark: '{extracted}'")
    assert "Capstone" in extracted, f"Expected 'Capstone' in extracted text, got '{extracted}'"

    manipulator = PixelManipulator()

    # 3. Test Multi-Percentage Simultaneous Batch (5%, 10%, 25%, 75%, 100%)
    print("\n--- Testing Multi-Percentage Simultaneous Batch (5%, 10%, 25%, 75%, 100%) ---")
    multi_res = manipulator.manipulate_multi_percentages(
        image_path=test_img_path,
        output_dir="uploads",
        percentages=[5.0, 10.0, 25.0, 75.0, 100.0],
        method='noise',
        intensity=50
    )
    assert multi_res['success'] == True, f"Multi-percentage failed: {multi_res.get('error')}"
    assert len(multi_res['tiers']) == 5, f"Expected 5 tiers, got {len(multi_res['tiers'])}"
    for t in multi_res['tiers']:
        print(f"[OK] Tier {t['percentage']}%: Manipulated {t['manipulated_count']} px | MAE={t['mae']} | PSNR={t['psnr']} dB | Watermark: '{t['extracted_text']}' (Intact={t['watermark_intact']})")

    # 4. Test Area / ROI Selection Manipulation
    print("\n--- Testing ROI Area Selection Manipulation ---")
    roi_box = {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50, 'w': 40, 'h': 40}  # 40x40 = 1600 pixels
    roi_res = manipulator.manipulate_image(
        image_path=test_img_path,
        output_path="uploads/test_roi_manip.png",
        percentage=50.0,
        method='noise',
        intensity=50,
        roi=roi_box
    )
    assert roi_res['success'] == True, f"ROI manipulation failed: {roi_res.get('error')}"
    expected_roi_manip = int(round(0.50 * 1600))  # 800 pixels
    assert roi_res['manipulated_count'] == expected_roi_manip, f"Expected {expected_roi_manip}, got {roi_res['manipulated_count']}"
    assert roi_res['roi'] is not None
    print(f"[OK] ROI Area Manipulation: Targeted {roi_res['manipulated_count']} pixels in 40x40 ROI (ROI Actual={roi_res['roi_actual_percentage']}%, Total Actual={roi_res['actual_percentage']}%)")

    # 5. Test SIPI Database URL Resolver & Grayscale Clock Preset
    print("\n--- Testing SIPI Database URL Resolver & Grayscale Clock Preset ---")
    # 5a. Direct SIPI URL import for image 13 (Clock)
    clock_url = "https://sipi.usc.edu/database/database.php?volume=misc&image=13#top"
    clock_target = "uploads/sipi_test_clock.png"
    import_res = sipi_manager.import_from_url(clock_url, clock_target)
    assert import_res['success'] == True, f"SIPI URL import failed: {import_res.get('error')}"
    assert os.path.exists(import_res['path']), f"File not created: {import_res['path']}"
    clock_path = import_res['path']
    print(f"[OK] SIPI URL Import ('{clock_url}') -> Saved to {clock_path}")

    # 5b. Embed and verify watermark in imported Clock benchmark
    wm.embed_in_lsb(clock_path, "Capstone")
    clock_wm = wm.extract_lsb(clock_path)
    assert "Capstone" in clock_wm, f"Expected watermark in Clock benchmark, got '{clock_wm}'"
    print(f"[OK] Clock benchmark watermark extracted: '{clock_wm}'")

    # 6. Test Grayscale Percentage Manipulation (Presets + Custom 37%, 48%, 99%)
    print("\n--- Testing Grayscale Percentage Manipulation ---")
    
    # 6a. Custom 37% Grayscale Noise
    g37_res = manipulator.manipulate_grayscale_percentage(
        image_path=clock_path,
        output_path="uploads/test_clock_gray_37pct.png",
        percentage=37.0,
        gray_method="luminance",
        op_type="gray_noise",
        intensity=45
    )
    assert g37_res['success'] == True, f"Grayscale 37% failed: {g37_res.get('error')}"
    total_clock_px = g37_res['total_pixels']
    expected_37_px = int(round(0.37 * total_clock_px))
    assert abs(g37_res['manipulated_count'] - expected_37_px) <= 2, f"Expected ~{expected_37_px} altered, got {g37_res['manipulated_count']}"
    print(f"[OK] Grayscale 37% Noise: Altered {g37_res['manipulated_count']} of {total_clock_px} pixels ({g37_res['actual_percentage']}%) | MAE={g37_res['mae']} | PSNR={g37_res['psnr']} dB")

    # 6b. Custom 48% with ROI Selection (e.g. Center 100x100 = 10,000 pixels)
    gray_roi = {'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200}  # 100x100 = 10,000 pixels
    g48_res = manipulator.manipulate_grayscale_percentage(
        image_path=clock_path,
        output_path="uploads/test_clock_gray_48pct_roi.png",
        percentage=48.0,
        gray_method="luminance",
        op_type="lsb_corrupt",
        intensity=1,
        roi=gray_roi
    )
    assert g48_res['success'] == True, f"Grayscale 48% with ROI failed: {g48_res.get('error')}"
    expected_roi_48_px = int(round(0.48 * 10000))  # 4800 pixels
    assert g48_res['manipulated_count'] == expected_roi_48_px, f"Expected exactly {expected_roi_48_px} pixels in ROI, got {g48_res['manipulated_count']}"
    assert g48_res['roi_actual_percentage'] == 48.0
    print(f"[OK] Grayscale 48% ROI (100x100): Exactly {g48_res['manipulated_count']} pixels altered in ROI ({g48_res['roi_actual_percentage']}%) | Watermark: '{g48_res['extracted_text']}'")

    # 6c. Custom 99% Grayscale Inversion
    g99_res = manipulator.manipulate_grayscale_percentage(
        image_path=clock_path,
        output_path="uploads/test_clock_gray_99pct.png",
        percentage=99.0,
        gray_method="luminance",
        op_type="gray_invert"
    )
    assert g99_res['success'] == True, f"Grayscale 99% failed: {g99_res.get('error')}"
    print(f"[OK] Grayscale 99% Invert: Altered {g99_res['manipulated_count']} pixels ({g99_res['actual_percentage']}%) | MAE={g99_res['mae']}")

    # 6d. Test all standard preset percentages (5%, 10%, 25%, 50%, 75%, 100%)
    for pct in [5.0, 10.0, 25.0, 50.0, 75.0, 100.0]:
        res = manipulator.manipulate_grayscale_percentage(
            image_path=clock_path,
            output_path=f"uploads/test_clock_gray_{int(pct)}pct.png",
            percentage=pct,
            gray_method="luminance",
            op_type="salt_pepper"
        )
        assert res['success'] == True, f"Preset {pct}% failed: {res.get('error')}"
        assert abs(res['actual_percentage'] - pct) < 0.2
        print(f"[OK] Grayscale Preset {pct}%: Altered {res['manipulated_count']} pixels ({res['actual_percentage']}%)")

    # 7. Test Grayscale Multi-Percentage Matrix (6 simultaneous tiers)
    print("\n--- Testing Grayscale 6-Tier Simultaneous Matrix ---")
    gray_multi_res = manipulator.manipulate_grayscale_multi_percentages(
        image_path=clock_path,
        output_dir="uploads",
        percentages=[5.0, 10.0, 25.0, 50.0, 75.0, 100.0],
        gray_method="luminance",
        op_type="gray_noise",
        intensity=40
    )
    assert gray_multi_res['success'] == True, f"Grayscale multi matrix failed: {gray_multi_res.get('error')}"
    assert len(gray_multi_res['tiers']) == 6, f"Expected 6 tiers, got {len(gray_multi_res['tiers'])}"
    for t in gray_multi_res['tiers']:
        print(f"[OK] Grayscale Matrix Tier {t['percentage']}%: {t['manipulated_count']} px | MAE={t['mae']} | PSNR={t['psnr']} dB | Status: {t['watermark_status']}")

    # 8. Test Grayscale Excel Statistical Export
    print("\n--- Testing Grayscale Excel Report Generation ---")
    gray_excel_path = "uploads/test_gray_manip_report.xlsx"
    gray_exp_res = manipulator.export_manipulation_excel(
        image_path=clock_path,
        manipulated_path="uploads/test_clock_gray_37pct.png",
        excel_path=gray_excel_path,
        stats_payload=g37_res
    )
    assert gray_exp_res['success'] == True, f"Grayscale Excel export failed: {gray_exp_res.get('error')}"
    assert os.path.exists(gray_excel_path)
    print(f"[OK] Grayscale Excel Report generated: {gray_excel_path} ({os.path.getsize(gray_excel_path):,} bytes)")

    # 9. Test Flask App Grayscale & SIPI API Endpoints
    print("\n--- Testing Flask App Routes & Grayscale Endpoints ---")
    client = app.test_client()
    
    with client.session_transaction() as sess:
        sess['logged_in'] = True
        sess['username'] = 'admin'

    # 9a. POST /api/sipi/import with URL
    resp_sipi_url = client.post("/api/sipi/import", json={
        "url": "https://sipi.usc.edu/database/database.php?volume=misc&image=13#top",
        "name": "sipi_clock_flask.png"
    })
    assert resp_sipi_url.status_code == 200, f"SIPI Import URL failed: {resp_sipi_url.status_code}"
    sipi_url_json = resp_sipi_url.get_json()
    assert sipi_url_json['success'] == True
    clock_flask_filename = sipi_url_json['filename']
    print(f"[OK] POST /api/sipi/import (URL) -> 200 OK (Imported as '{clock_flask_filename}')")

    # 9b. POST /api/pixel-manipulate-gray-percentage/<filename> with custom 37%
    resp_g37 = client.post(f"/api/pixel-manipulate-gray-percentage/{clock_flask_filename}", json={
        "percentage": 37.0,
        "gray_method": "luminance",
        "op_type": "gray_noise",
        "intensity": 50
    })
    assert resp_g37.status_code == 200, f"API 37% error: {resp_g37.status_code}"
    g37_json = resp_g37.get_json()
    assert g37_json['success'] == True
    assert g37_json['percentage'] == 37.0
    print(f"[OK] POST /api/pixel-manipulate-gray-percentage (37%) -> 200 OK (Altered {g37_json['manipulated_count']} pixels)")

    # 9c. POST /api/pixel-manipulate-gray-percentage/<filename> with custom 48% + ROI
    resp_g48 = client.post(f"/api/pixel-manipulate-gray-percentage/{clock_flask_filename}", json={
        "percentage": 48.0,
        "gray_method": "luminance",
        "op_type": "lsb_corrupt",
        "roi": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
    })
    assert resp_g48.status_code == 200, f"API 48% error: {resp_g48.status_code}"
    g48_json = resp_g48.get_json()
    assert g48_json['success'] == True
    assert g48_json['roi_actual_percentage'] == 48.0
    print(f"[OK] POST /api/pixel-manipulate-gray-percentage (48% in ROI) -> 200 OK (Altered {g48_json['manipulated_count']} pixels)")

    # 9d. POST /api/pixel-manipulate-gray-percentage/<filename> with custom 99%
    resp_g99 = client.post(f"/api/pixel-manipulate-gray-percentage/{clock_flask_filename}", json={
        "percentage": 99.0,
        "gray_method": "luminance",
        "op_type": "gray_invert"
    })
    assert resp_g99.status_code == 200, f"API 99% error: {resp_g99.status_code}"
    g99_json = resp_g99.get_json()
    assert g99_json['success'] == True
    assert g99_json['actual_percentage'] == 99.0
    print(f"[OK] POST /api/pixel-manipulate-gray-percentage (99%) -> 200 OK (Altered {g99_json['manipulated_count']} pixels)")

    # 9e. POST /api/pixel-manipulate-gray-multi/<filename>
    resp_gmulti = client.post(f"/api/pixel-manipulate-gray-multi/{clock_flask_filename}", json={
        "percentages": [5.0, 10.0, 25.0, 50.0, 75.0, 100.0],
        "gray_method": "luminance",
        "op_type": "gray_noise",
        "intensity": 50
    })
    assert resp_gmulti.status_code == 200, f"API gray multi error: {resp_gmulti.status_code}"
    gmulti_json = resp_gmulti.get_json()
    assert gmulti_json['success'] == True
    assert len(gmulti_json['tiers']) == 6
    print(f"[OK] POST /api/pixel-manipulate-gray-multi -> 200 OK (returned {len(gmulti_json['tiers'])} tiers simultaneously)")

    # 9f. GET /export-manipulation-excel with Grayscale percentage query
    resp_gexp = client.get(f"/export-manipulation-excel/{clock_flask_filename}?is_gray_percentage=true&percentage=37&gray_method=luminance&op_type=gray_noise&intensity=50")
    assert resp_gexp.status_code == 200, f"Excel download error: {resp_gexp.status_code}"
    print(f"[OK] GET /export-manipulation-excel (Grayscale % report) -> 200 OK ({len(resp_gexp.data):,} bytes)")

    # 10. Test Pixel Block Color & 0-255 RGB Byte Manipulation
    print("\n--- Testing Custom Pixel Block Color & Byte Manipulation (0-255 Range) ---")
    
    # 10a. Inspect pixel at specific coordinate
    resp_pixel = client.get(f"/api/pixel-at-coord/{clock_flask_filename}?x=20&y=30")
    assert resp_pixel.status_code == 200
    pixel_data = resp_pixel.get_json()
    assert pixel_data['success'] == True
    assert pixel_data['pixel']['x'] == 20
    assert pixel_data['pixel']['y'] == 30
    assert 0 <= pixel_data['pixel']['r'] <= 255
    assert 0 <= pixel_data['pixel']['g'] <= 255
    assert 0 <= pixel_data['pixel']['b'] <= 255
    print(f"[OK] GET /api/pixel-at-coord -> 200 OK (Inspected (20,30) RGB=({pixel_data['pixel']['r']},{pixel_data['pixel']['g']},{pixel_data['pixel']['b']}))")

    # 10b. Custom byte manipulation with multiple pixel blocks and custom 0-255 byte values
    resp_block = client.post(f"/api/pixel-custom-byte-manipulate/{clock_flask_filename}", json={
        "blocks": [
            {"x": 10, "y": 10, "w": 8, "h": 8, "r": 255, "g": 0, "b": 128, "hex": "#ff0080"},
            {"x": 40, "y": 50, "w": 16, "h": 16, "r": 0, "g": 255, "b": 64, "hex": "#00ff40"},
            {"x": 100, "y": 100, "w": 1, "h": 1, "r": 200, "g": 200, "b": 200, "hex": "#c8c8c8"}
        ],
        "block_size": 8,
        "r_val": 255,
        "g_val": 0,
        "b_val": 128,
        "target_channel": "all",
        "byte_op_type": "direct_replace"
    })
    assert resp_block.status_code == 200
    block_json = resp_block.get_json()
    assert block_json['status'] == 'success'
    expected_count = (8 * 8) + (16 * 16) + (1 * 1)
    assert block_json['stats']['manipulated_pixels'] == expected_count
    assert block_json['stats']['psnr'] > 0
    print(f"[OK] POST /api/pixel-custom-byte-manipulate (3 Blocks = {expected_count} px) -> 200 OK | PSNR={block_json['stats']['psnr']} dB")

    # 10c. Verify Excel export for custom byte blocks
    resp_block_exp = client.get(f"/export-manipulation-excel/{clock_flask_filename}?is_custom_byte=true&r_val=255&g_val=0&b_val=128&target_channel=all&byte_op_type=direct_replace")
    assert resp_block_exp.status_code == 200
    print(f"[OK] GET /export-manipulation-excel (Custom Block Byte report) -> 200 OK ({len(resp_block_exp.data):,} bytes)")

    # 11. Test Pixel Block Exchange & Percentage Manipulation Swapping
    print("\n--- Testing Pixel Block Exchange & Crop Slot Swap Suite ---")
    resp_exchange = client.post(f"/api/pixel-exchange/{clock_flask_filename}", json={
        "block_a": {"x": 10, "y": 10, "w": 16, "h": 16},
        "block_b": {"x": 60, "y": 60, "w": 16, "h": 16},
        "mode": "swap",
        "percentage": 25.0,
        "method": "noise",
        "intensity": 60
    })
    assert resp_exchange.status_code == 200
    exchange_json = resp_exchange.get_json()
    assert exchange_json['status'] == 'success'
    assert exchange_json['stats']['manipulated_pixels'] > 0
    print(f"[OK] POST /api/pixel-exchange (Swap 16x16 with 25% noise) -> 200 OK | Altered={exchange_json['stats']['manipulated_pixels']} px | PSNR={exchange_json['stats']['psnr']} dB")

    # 11b. Verify Excel export for pixel block exchange
    resp_exch_exp = client.get(f"/export-manipulation-excel/{clock_flask_filename}?is_exchange=true&mode=swap&ax1=10&ay1=10&aw=16&ah=16&bx1=60&by1=60&bw=16&bh=16&percentage=25&method=noise")
    assert resp_exch_exp.status_code == 200
    print(f"[OK] GET /export-manipulation-excel (Pixel Exchange report) -> 200 OK ({len(resp_exch_exp.data):,} bytes)")

    print("\n=== ALL ADVANCED PIXEL MANIPULATION, GRAYSCALE, CUSTOM BYTE & PIXEL BLOCK EXCHANGE TESTS PASSED WITH 0 ERRORS! ===")

if __name__ == "__main__":
    run_tests()

