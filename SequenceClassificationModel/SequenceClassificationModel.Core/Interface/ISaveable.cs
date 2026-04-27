namespace SequenceClassificationModel.Core.Interface
{
    public interface ISaveable
    {
        void Save(string path);
        void Load(string path);
    }
}