from torch import nn

class CNNNetworkIF(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64, 
                kernel_size=2,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        ) 
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1,
        #         padding=2
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(kernel_size=2)
        # )
        
        self.dropout = nn.Dropout(p=0.50)
        self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(13376, 4224) # hidden layer, ref : https://www.isca-archive.org/interspeech_2017/vasquezcorrea17_interspeech.pdf
        # ref : https://www.semanticscholar.org/paper/Temporal-Envelope-and-Fine-Structure-Cues-for-Using-Kodrasi/7f89e17b2328fb5cf2f037996637db05a3f3c416
        # self.linear2 = nn.Linear(1024, 2)

        self.fc = nn.Linear(4224, 2) # (in features and out features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        #print(input_data.shape)
        x = self.conv1(input_data)
        x = self.conv2(x)
        # x = self.conv3(x)
        
        x = self.dropout(x)
        x = self.flatten(x)
        #print(x.shape)
        #x = self.fc1(x)
        #x = self.softmax(x)
        
        #x = self.fc2(x)
        #x = self.softmax(x)

        # x = self.linear1(x)
        # pass through the ReLU 
        # x = nn.functional.relu(x)
        # logits = self.linear2(x)
        # predictions = self.softmax(logits)
        
        logits = self.fc(x)
        # predictions = self.softmax(logits)
        predictions = logits # no need to apply softmax as it is already applied in the loss function (CrossEntropyLoss)

        return predictions
    

