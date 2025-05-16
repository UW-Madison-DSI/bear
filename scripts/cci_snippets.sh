# Manually resize a partition and extend a logical volume

# List block devices and show details for sda
lsblk | grep -A 2 sda 

# Display free space in all volume groups
vgdisplay | grep Free  

# Resize partition 3 on /dev/sda
growpart /dev/sda 3  

# Extend the logical volume to use all free space
/sbin/lvextend --extents +100%FREE -r /dev/mapper/Volume00-home  