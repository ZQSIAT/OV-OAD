# OV-OAD
Este repositorio representa el primer paso hacia el aprovechamiento del aprendizaje basado en texto para la detección de acciones en línea sin necesidad de supervisión humana explícita.
<p align="center" width="100%">
<img src="https://i.postimg.cc/4dMR8CCY/overall.png" width="80%" height="80%">
</p>

La comprensión de videos depende de una detección precisa de acciones para realizar análisis temporales. Sin embargo, los métodos convencionales actuales presentan limitaciones en aplicaciones del mundo real debido a sus enfoques de evaluación offline y de conjunto cerrado, así como a su dependencia de anotaciones manuales. Para abordar estos desafíos y permitir la comprensión de acciones en tiempo real en entornos de mundo abierto, proponemos OV-OAD, un detector de acciones en línea sin supervisión que aprovecha los modelos de visión-lenguaje y aprende únicamente a partir de supervisión textual. Al incorporar una unidad decodificadora centrada en objetos en un modelo basado en Transformer, agrupamos fotogramas con semánticas similares utilizando la correspondencia entre video y texto.
La realización de experimentos exhaustivos en cuatro benchmarks de detección de acciones demuestra que OV-OAD supera a otros métodos avanzados de detección sin supervisión. En concreto, alcanza una precisión media del 37,5 % en THUMOS'14 y una precisión media calibrada del 73,8 % en TVSeries. Esta investigación establece una base sólida para la transferencia sin supervisión en la detección de acciones en línea, permitiendo soluciones escalables para la comprensión temporal en el mundo abierto. El código estará disponible para su descarga en https://github.com/OpenGVLab/OV-OAD.

## Notas de la versión
- **[10/07/2026] Lanzamiento del modelo OV-OAD y de los scripts de inferencia.🔥**
- **[04/10/2024] Lanzamiento del código de OV-OAD.**
- **[11/12/2024] Procesamiento de datos de OV-OAD de código abierto.**
- **[13/12/2024] Lanzamiento del código de entrenamiento y prueba de OV-OAD.**

## Modelos y scripts
### Modelo
Nuestro modelo está disponible como código abierto en [Google Drive](https://drive.google.com/file/d/1fxf3rZE8usKJJ0taCPgWboPrTcYFUtl1/view?usp=sharing). ¡Que lo disfrutes!

### Inferencia
```bash
vim configs/enc32_anet_3layers_lsxattn_only_inference.yml # Ajusta el directorio de archivos según tus necesidades
bash scripts/thumos_infer_only.sh
```

### Instalación

#### 1. **Clona este repositorio y navega hasta la carpeta OV-OAD:**
```bash
git clone https://github.com/OV-OAD
cd OV-OAD
```

#### 2. **Instala el paquete de inferencia:**
```bash
conda create -n ovoad python=3.7 -y
conda activate ovoad
```

### Navegación del proyecto
Para obtener más detalles sobre la inferencia y la evaluación, consulta la siguiente página.
#### Procesamiento de datos
```bash
bash extract_features/run_save_imgs.sh
bash extract_features/run_extc_feat.sh
```

#### Scripts de preentrenamiento
```bash
python -u -m main_pretrain \
    --cfg configs/enc32_anet_3layers_lsxattn.yml \
```

#### Script de evaluación
```bash
python -u -m main_pretrain \
    --cfg configs/test_oad.yml \
    --resume xxx/ovoador_bs256x1/best_map.pth \
```

## Citación

Si consideras que este trabajo es útil para tu investigación y aplicaciones, por favor cita los artículos/blogs relacionados utilizando este BibTeX:
```bibtex
@article{zhao2024ovoad,
  title={¿Ayuda el preentrenamiento video-texto a la detección de acciones en línea de vocabulario abierto?},
  author={Qingsong Zhao, Yi Wang, Jilan Xu, Yinan He, Zifan Song, Limin Wang, Yu Qiao, Cairong Zhao},
  journal={Advances in neural information processing systems},
  year={2024}
}
```

## Agradecimiento

- [GroupViT](https://github.com/groupvit): el código base en el que nos basamos.
- [Uniformer](https://github.com/uniformer): el código base en el que nos basamos.
