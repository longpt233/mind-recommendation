# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch.nn.functional as F

__all__ = ["NRMSModelEntity"]

from reco_utils.recommender.newsrec.models.nrms import NRMSModel


class NRMSModelEntity(NRMSModel):

    def __init__(self, hparams):
        super().__init__(hparams)

    def news_encoder(self, sequences_input):
        title, entity = sequences_input[:, :self.hparams.title_size], sequences_input[:, self.hparams.title_size:]
        y = F.dropout(self.embedding_layer(title), p=self.hparams.dropout).transpose(0, 1)
        q = F.dropout(self.embedding_layer(entity), p=self.hparams.dropout).transpose(0, 1)
        # y = self.news_self_att(q, y, y)[0].transpose(0, 1)

        # 2 att 
        y = self.news_self_att(q, y, y)[0]
        y = self.news_self_att_2(y, y, y)[0]  # số 0 này là lấy cái đầu cho vui thôi k có gì 
        y = F.dropout(y, p=self.hparams.dropout).transpose(0, 1)


        # y = F.dropout(y, p=self.hparams.dropout)
        y = self.news_att_layer(y)
        return y


