import os
import sys

# Try to find the project root automatically
# First check if we're in the tre_code directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'tre_code' or current_dir.endswith('/tre_code'):
    project_root = current_dir + '/'
    local_pc_root = project_root  # Keep for backward compatibility
else:
    # Fall back to original behavior
    local_pc_root = os.path.expanduser("~/tre_code/")
    if os.path.isdir(local_pc_root):
        project_root = local_pc_root
    else:
        project_root = os.path.expanduser("~/")
        local_pc_root = project_root  # Keep for backward compatibility

density_data_root = os.path.expanduser(project_root + 'density_data/')
utils_root = os.path.expanduser(project_root + 'utils/')

roots = [project_root, density_data_root, utils_root]
for root in roots:
    if root not in sys.path:
        sys.path.append(root)
