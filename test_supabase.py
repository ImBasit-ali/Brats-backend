import supabase
import os

client = supabase.create_client('https://abc.supabase.co', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFiYyIsInJvbGUiOiJhbm9uIiwiaWF0IjoxNjE2MDMzNzc1LCJleHAiOjE5MzE2MDk3NzV9.xyz')
try:
    res = client.storage.from_('test').create_signed_upload_url('test_path.nii')
    print("RES:", res)
except Exception as e:
    print("ERR:", e)
