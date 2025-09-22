import numpy as np
import pandas as pd

num_samples = 1000

# Ocean floor images
X_images = np.random.rand(num_samples, 64, 64, 1)
np.save('ocean_images.npy', X_images)

# Tabular historical/astronomical features
X_tabular = pd.DataFrame({
    'year_of_event': np.random.randint(1000, 2000, num_samples),
    'moon_phase': np.random.rand(num_samples),
    'solstice_distance': np.random.rand(num_samples)*180,
    'known_site_distance': np.random.rand(num_samples)*50
})

# Synthetic label
y = ((X_tabular['moon_phase'] > 0.6) &
     (X_tabular['solstice_distance'] < 90) &
     (X_tabular['known_site_distance'] < 25)).astype(int)
X_tabular['label'] = y

X_tabular.to_csv('historical_data.csv', index=False)
print("Datasets created: ocean_images.npy, historical_data.csv")
