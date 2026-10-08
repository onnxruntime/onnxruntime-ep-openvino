# OVEP Bin Utils Tool

This unified C++ utility provides comprehensive operations for OpenVINO™ Execution Provider (EP) binary cache files (`.bin`):
- **Extract**: Extract blobs and metadata from existing bin files
- **Generate**: Create bin files from compiled model blob files

## Usage

### Extract blobs from a bin file:

```bash
# Extract to default directory (<input_stem>_extract/)
ovep_bin_utils extract model_OpenVINOExecutionProvider.bin

# Extract to specific directory
ovep_bin_utils extract model.bin --out ./output
```

**Output:**
- `bin_utils.json` - Simplified blob mapping with relative paths (for regeneration)
- `metadata.json` - Full BSON metadata in JSON format (for reference)
- `<blob_name>.blob` - One file per blob

### Generate bin file from blob(s):

```bash
# Single blob
ovep_bin_utils generate --blob model.blob --name partition_0 --output model.bin

# Multiple blobs (multi-partition models)
ovep_bin_utils generate \
  --blob model1.blob --name partition_0 \
  --blob model2.blob --name partition_1 \
  --output model_OpenVINOExecutionProvider.bin

# Generate from extracted bin_utils.json (automatically finds .blob files)
ovep_bin_utils generate --metadata bin_utils.json --output model.bin
```
