import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd

def hitung_mape():
    try:
        # Ambil input dari Entry
        gt_input = entry_gt.get()
        pred_input = entry_pred.get()

        # Convert ke array angka
        luas_gt = np.array([float(i) for i in gt_input.split(',')])
        luas_pred = np.array([float(i) for i in pred_input.split(',')])

        # Validasi panjang
        if len(luas_gt) != len(luas_pred):
            messagebox.showerror("Error", "Jumlah nilai pada kedua input harus sama!")
            return

        epsilon = 1e-10
        mape = np.mean(np.abs((luas_gt - luas_pred) / (luas_gt + epsilon))) * 100

        # Buat DataFrame
        data = {
            'Luas Manual (cm²)': luas_gt,
            'Luas Model (cm²)': luas_pred,
            'Error (%)': np.abs((luas_gt - luas_pred) / (luas_gt + epsilon)) * 100
        }
        df = pd.DataFrame(data)

        # Tampilkan hasil di text box
        text_output.delete('1.0', tk.END)
        text_output.insert(tk.END, df.to_string(index=False))
        text_output.insert(tk.END, f"\n\nNilai MAPE: {mape:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

# Inisialisasi GUI
root = tk.Tk()
root.title("Hitung MAPE Luas Lubang Jalan")

# Label dan Entry GT
label_gt = tk.Label(root, text="Luas Manual (cm²), pisahkan dengan koma:")
label_gt.pack()
entry_gt = tk.Entry(root, width=60)
entry_gt.pack()

# Label dan Entry Prediksi
label_pred = tk.Label(root, text="Luas Model (cm²), pisahkan dengan koma:")
label_pred.pack()
entry_pred = tk.Entry(root, width=60)
entry_pred.pack()

# Tombol Hitung
btn_hitung = tk.Button(root, text="Hitung MAPE", command=hitung_mape)
btn_hitung.pack(pady=10)

# Output hasil
text_output = tk.Text(root, height=15, width=80)
text_output.pack()

# Jalankan aplikasi
root.mainloop()