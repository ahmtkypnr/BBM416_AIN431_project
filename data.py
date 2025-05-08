import kagglehub
import os
import random
import shutil
from tqdm import tqdm
import csv
import json
import yaml


def create_txt(txt_path, lines=None):
    # TXT dosyasını oluşturma
    with open(txt_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def parse_yolo_label_csv(csv_path,classes_dict):
    classes_list = list(classes_dict)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        lines = []
        classes = {}
        for row in reader:
            img_width = int(row['width'])
            img_height = int(row['height'])
            # BBox koordinatları
            xmin = int(row['xmin'])
            xmax = int(row['xmax'])
            ymin = int(row['ymin'])
            ymax = int(row['ymax'])

            x_center = float((xmin + xmax) / 2)/img_width
            y_center = float((ymin + ymax) / 2)/img_height
            width = float(xmax - xmin)/img_width
            height = float(ymax - ymin)/img_height
            box_class = classes_list.index(row['class'])

            lines.append(" ".join([str(box_class), str(x_center), str(y_center), str(width), str(height)]))

        return lines
def get_class(csv_path):
    # CSV dosyasını oku ve sınıfları al
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        classes = {}
        for row in reader:
            box_class = row['class']
            if box_class not in classes:
                classes[box_class] = 1
            else:
                classes[box_class] += 1
    return max(classes, key=classes.get)

def exclude_classes(classes_to_exclude: list, dataset_path=os.getcwd()):

    json_path = os.path.join(dataset_path, 'classes.json')

    try:
        # Dosyayı aç ve içeriği dictionary olarak yükle
        with open(json_path, 'r') as json_file:
            classes = json.load(json_file)
        print("Dosya yüklendi.")

    except FileNotFoundError:
        print(f"{json_path} bulunamadı, yeni bir dosya oluşturulacak.")
        
        # Eğer dosya yoksa, yeni bir dictionary oluştur ve dosyaya kaydet
        classes = {}
        csvs = []  # csvs değişkeni burada tanımlanmalı
        # Burada csvs listesini uygun şekilde doldurmalısınız, örneğin:
        # csvs = [os.path.join(dataset_path, 'csv_files', f) for f in os.listdir(os.path.join(dataset_path, 'csv_files'))]
        with tqdm(csvs, total=len(csvs)) as progress_bar:
            for csv_path in progress_bar:
                name = csv_path.split(os.sep)[-1].replace(".csv", "")
                _class = get_class(csv_path)  # get_class fonksiyonunun nasıl çalıştığını doğrulamalısınız
                if _class not in classes:
                    classes[_class] = []
                classes[_class].append(name)

        # JSON dosyasını oluştur ve dictionary'i yaz
        with open(json_path, 'w') as json_file:
            json.dump(classes, json_file, indent=4)
        
        print(f"{json_path} başarıyla oluşturuldu.")

    except json.JSONDecodeError:
        print(f"{json_path} hatalı bir JSON formatına sahip, dosya okunamadı.")
        return

    # 'classes_to_exclude' parametresine göre sınıfları dışarıda bırak
    for class_to_exclude in classes_to_exclude:
        if class_to_exclude in classes:
            del classes[class_to_exclude]

    # Sadece dahil edilen sınıfları ekle
    classes_to_include = list(classes.keys())

    data = {
      'path': dataset_path,  # dataset root directory
      'train': 'images/train',  # train images folder
      'val': 'images/val',  # validation images folder
      'test': 'images/test',  # test images folder

      'nc': len(classes_to_include),
      'names': classes_to_include
    }

    with open(os.path.join(dataset_path, 'data_excluded.yaml'), 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def create_shuffle_dataset(dataset_path = os.getcwd(),train = 0.8,val=0.1,colab_mode = True): # /content for colab # the rest will be reserved for test set
    """
    dataseti indirir, oluştruru istenen fomrata geitirir ve train, val, test setlerine ayırır.
    dataset_path: datasetin kaydedileceği dizin
    colab_mode: colabda çalışıyorsan True, bilgisayarında çalışıyorsan False yap, dataseti iki kere kopyalar, datasetin yedeği kalması için true da tutabilirsin
    """
    path = kagglehub.dataset_download("a2015003713/militaryaircraftdetectiondataset")
    path = os.path.join(path,"dataset")
    #dataseti oluşturacağımız dizin
    dataset_path = os.path.join(dataset_path ,'dataset')

    all_data_path = os.path.join(dataset_path, "all")

    #gerekli dizinleri oluştur
    os.makedirs(dataset_path, exist_ok=True)

    os.makedirs(all_data_path, exist_ok=True)
    if len(os.listdir(all_data_path)) == 0:
        print("Dataset dizine aktarılıyor...")
        for item in tqdm(os.listdir(path)):
            s = os.path.join(path, item)
            d = os.path.join(all_data_path,item)
            if colab_mode:

                shutil.copy(s, d) # colabde çalışmıyorsan, bilgisayarında yer kaplamaması için burayı shutil.move olarak değiştirebilirsin, datayı bir yere yedeklemeni öneririm
            else:
                shutil.move(s, d) 
    else:
        print("Dataset zaten mevcut, tekrar aktarılmayacak.")

    print("Dataset boyutu: ",len(os.listdir( all_data_path)))

    csvs = [os.path.join(all_data_path,f) for f in os.listdir(all_data_path) if f.endswith('.csv')]

        # Dosyanın mevcut olup olmadığını kontrol et
    json_path = os.path.join(dataset_path,'classes.json')

    try:
        # Dosyayı aç ve içeriği dictionary olarak yükle
        with open(json_path, 'r') as json_file:
            classes = json.load(json_file)
        print("Dosya yüklendi.")

    except FileNotFoundError:
        print(f"{json_path} bulunamadı, yeni bir dosya oluşturulacak.")
        
        # Eğer dosya yoksa, yeni bir dictionary oluştur ve dosyaya kaydet
        classes = {}
        with tqdm(csvs, total=len(csvs)) as progress_bar:
            for csv_path in progress_bar:
                name = csv_path.split(os.sep)[-1].replace(".csv", "")
                _class = get_class(csv_path)
                if _class not in classes:
                    classes[_class] = []
                classes[_class].append(name)
        # JSON dosyasını oluştur ve dictionary'i yaz
        with open(json_path, 'w') as json_file:
            json.dump(classes, json_file, indent=4)
        
        print(f"{json_path} başarıyla oluşturuldu.")

    except json.JSONDecodeError:
        print(f"{json_path} hatalı bir JSON formatına sahip, dosya okunamadı.")
        # Eski dosylar silinip yeni klasörler oluşturuluyor
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    if os.path.exists(images_path):
        shutil.rmtree(images_path)
        print(f"{images_path} klasörü silindi.")
    else:
        print(f"{images_path} klasörü yok. (Sorun değil!)")

    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
        print(f"{labels_path} klasörü silindi.")
    else:
        print(f"{labels_path} klasörü yok. (Sorun değil!)")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_path, split), exist_ok=True)
        os.makedirs(os.path.join(labels_path, split), exist_ok=True)

    #label dataları ve classes json'u oluştur
    print("txt'ler oluşturuluyor")
    existing_files = set(os.listdir(all_data_path))
    for csv_path in tqdm(csvs):
        if csv_path.replace('.csv', '.txt') in existing_files:
            continue
        try:
            lines = parse_yolo_label_csv(csv_path,classes)
        except Exception as e:
            print(f"Error parsing {csv_path}: {e}")
            continue
        csv_path = csv_path.replace('.csv', '.txt')
        create_txt(csv_path,lines)

    # Dosya listesi al
    images = [f for f in os.listdir(all_data_path) if f.endswith('.jpg')]
    labels = [f for f in os.listdir(all_data_path) if f.endswith('.txt')]

    images.sort()  # eşleşme garantisi için
    labels.sort()

    # Shuffle
    random.seed(42)
    random.shuffle(images)

    # %80 train, %10 val %10 test
    split_idx = int(train * len(images))
    val_idx = int(val * len(images))

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:split_idx+val_idx]
    test_imgs = images[split_idx+val_idx:]

    # Hedef klasörler
    for split, img_list in zip(['train', 'val','test'], [train_imgs, val_imgs,test_imgs]):
        print(split)
        for img_name in tqdm(img_list):
            # image taşı
            shutil.copy(os.path.join(all_data_path ,img_name), os.path.join(dataset_path,'images',split,img_name) )
            # label taşı
            label_name = img_name.replace('.jpg', '.txt')
            shutil.copy(os.path.join(all_data_path ,label_name), os.path.join(dataset_path,'labels',split,label_name) )
    data = {
    'path': dataset_path,  # dataset root directory
    'train': 'images/train',  # train images folder
    'val': 'images/val',  # validation images folder
    'test': 'images/test',  # test images folder

    'nc': len(list(classes.keys())),
    'names': list(classes.keys())
    }

    with open(os.path.join(dataset_path,'data.yaml'), 'w') as f:
        yaml.dump(data, f, sort_keys=False)