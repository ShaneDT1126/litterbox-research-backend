import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load environment variables from .env file
load_dotenv()

# Get connection string from environment variable
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "litterbox-vectorstore")

if not connection_string:
    print("Error: AZURE_STORAGE_CONNECTION_STRING environment variable not set")
    exit(1)

try:
    # Create blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Get container client
    container_client = blob_service_client.get_container_client(container_name)

    # Check if container exists
    container_exists = container_client.exists()

    if container_exists:
        print(f"✅ Successfully connected to container '{container_name}'")

        # List blobs in container
        blobs = list(container_client.list_blobs())
        print(f"Found {len(blobs)} blobs in container")

        # Create a test blob
        test_blob_name = "test-blob.txt"
        test_blob_client = container_client.get_blob_client(test_blob_name)
        test_blob_client.upload_blob("This is a test blob", overwrite=True)
        print(f"✅ Successfully uploaded test blob '{test_blob_name}'")

        # Download the test blob
        download_stream = test_blob_client.download_blob()
        content = download_stream.readall().decode("utf-8")
        print(f"✅ Successfully downloaded test blob with content: '{content}'")

        # Delete the test blob
        test_blob_client.delete_blob()
        print(f"✅ Successfully deleted test blob '{test_blob_name}'")
    else:
        print(f"❌ Container '{container_name}' does not exist")

except Exception as e:
    print(f"❌ Error: {str(e)}")